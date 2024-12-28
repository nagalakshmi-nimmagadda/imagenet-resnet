import os
import requests
import json
from pathlib import Path
import shutil
import subprocess
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()  # Load environment variables from .env file

def get_huggingface_username():
    """Get HuggingFace username from environment or prompt user."""
    username = os.getenv('HUGGINGFACE_USERNAME')
    if not username:
        username = input("Enter your HuggingFace username: ").strip()
        # Save to .env file for future use
        with open('.env', 'a') as f:
            f.write(f'\nHUGGINGFACE_USERNAME={username}')
    return username

def install_git_lfs():
    """Install git-lfs if not already installed."""
    try:
        subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
        print("git-lfs is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing git-lfs...")
        try:
            # For Ubuntu/Debian
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "git-lfs"], check=True)
            subprocess.run(["git", "lfs", "install"], check=True)
            print("git-lfs installed successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to install git-lfs: {e}")

def download_example_images():
    """Download example images."""
    # Using stable public image URLs
    examples = {
        "cat.jpg": [
            "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg",
            "https://cdn.pixabay.com/photo/2017/02/20/18/03/cat-2083492_1280.jpg"
        ],
        "dog.jpg": [
            "https://upload.wikimedia.org/wikipedia/commons/2/2d/Dog_-_Canis_lupus_familiaris.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/8/8c/Poligraf_Poligrafovich.jpg",
            "https://cdn.pixabay.com/photo/2016/12/13/05/15/puppy-1903313_1280.jpg"
        ],
        "bird.jpg": [
            "https://upload.wikimedia.org/wikipedia/commons/3/32/House_sparrow04.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/a/ae/Carduelis_carduelis_close_up.jpg",
            "https://cdn.pixabay.com/photo/2017/02/07/16/47/kingfisher-2046453_1280.jpg"
        ]
    }
    
    # Backup URLs using different CDNs
    backup_images = {
        "cat.jpg": "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg",
        "dog.jpg": "https://images.pexels.com/photos/1805164/pexels-photo-1805164.jpeg",
        "bird.jpg": "https://images.pexels.com/photos/349758/hummingbird-bird-birds-349758.jpeg"
    }
    
    Path("examples").mkdir(exist_ok=True)
    
    def resize_image(image_path, size=(224, 224)):
        """Resize image to target size."""
        from PIL import Image
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Resize maintaining aspect ratio
                img.thumbnail(size, Image.Resampling.LANCZOS)
                # Create new image with padding if needed
                new_img = Image.new('RGB', size, (255, 255, 255))
                # Paste resized image in center
                offset = ((size[0] - img.size[0]) // 2,
                         (size[1] - img.size[1]) // 2)
                new_img.paste(img, offset)
                new_img.save(image_path)
                print(f"Resized {image_path} to {size}")
        except Exception as e:
            print(f"Error resizing image {image_path}: {e}")
    
    for name, urls in examples.items():
        success = False
        temp_path = f"examples/temp_{name}"
        final_path = f"examples/{name}"
        
        # Try primary sources
        for url in urls:
            try:
                print(f"Trying to download {name} from {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Save to temporary file first
                with open(temp_path, "wb") as f:
                    f.write(response.content)
                
                # Validate and resize image
                resize_image(temp_path)
                
                # Move to final location
                shutil.move(temp_path, final_path)
                print(f"Successfully downloaded and processed {name}")
                success = True
                break
            except Exception as e:
                print(f"Failed attempt: {e}")
                if Path(temp_path).exists():
                    Path(temp_path).unlink()
                continue
        
        # Try backup source if primary sources failed
        if not success:
            try:
                print(f"Trying backup source for {name}")
                response = requests.get(backup_images[name], timeout=10)
                response.raise_for_status()
                
                with open(temp_path, "wb") as f:
                    f.write(response.content)
                
                resize_image(temp_path)
                shutil.move(temp_path, final_path)
                print(f"Successfully downloaded {name} from backup source")
                success = True
            except Exception as e:
                print(f"Backup source failed: {e}")
                if Path(temp_path).exists():
                    Path(temp_path).unlink()
        
        # Create placeholder if all downloads failed
        if not success:
            try:
                from PIL import Image, ImageDraw, ImageFont
                img = Image.new('RGB', (224, 224), color='white')
                draw = ImageDraw.Draw(img)
                draw.text((112, 112), name.split('.')[0], fill='black', anchor="mm")
                img.save(final_path)
                print(f"Created placeholder image for {name}")
            except Exception as e:
                print(f"Failed to create placeholder: {e}")
    
    # Clean up any temporary files
    for temp_file in Path("examples").glob("temp_*"):
        temp_file.unlink()
    
    print("Example images preparation complete!")

def download_imagenet_classes():
    """Download ImageNet class labels."""
    # Try to download the complete set of ImageNet labels
    urls = [
        "https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/meta/imagenet_classes.txt",
        "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt",
        "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    ]
    
    classes = None
    for url in urls:
        try:
            print(f"Downloading ImageNet classes from {url}")
            response = requests.get(url)
            response.raise_for_status()
            
            if url.endswith('.json'):
                classes = response.json()
            elif url.endswith('.txt'):
                if 'clsidx_to_labels' in url:
                    classes = eval(response.text)
                    classes = {str(k): v for k, v in classes.items()}
                else:
                    classes = {str(i): label.strip() for i, label in enumerate(response.text.splitlines())}
            
            # Verify we have all 1000 classes
            if len(classes) >= 1000:
                break
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue
    
    # If no complete set was found, create a basic set
    if not classes or len(classes) < 1000:
        print("Creating basic class labels...")
        classes = {str(i): f"Class {i}" for i in range(1000)}
    
    # Save to file
    with open("imagenet_classes.json", "w") as f:
        json.dump(classes, f, indent=2, sort_keys=True)
    print(f"Saved {len(classes)} ImageNet classes")

def copy_latest_checkpoint():
    """Copy the latest checkpoint."""
    src = Path("../checkpoints/last.ckpt")
    if not src.exists():
        raise FileNotFoundError(f"No checkpoint found at {src}!")
    
    shutil.copy(src, "model.ckpt")
    print("Copied latest checkpoint")

def verify_huggingface_token():
    """Verify HuggingFace token and permissions."""
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("\nNo HuggingFace token found. Please follow these steps:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Create a new token with 'write' access")
        print("3. Enter the token below\n")
        token = input("Enter your HuggingFace token: ").strip()
        # Save to .env file
        with open('.env', 'a') as f:
            f.write(f'\nHUGGINGFACE_TOKEN={token}\n')
    
    # Verify token
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info['name']
        print(f"Successfully authenticated as: {username}")
        return username, token
    except Exception as e:
        print("Error verifying token. Please ensure:")
        print("1. The token is valid")
        print("2. The token has 'write' permissions")
        print("3. You are connected to the internet")
        raise Exception(f"Authentication failed: {str(e)}")

def create_huggingface_space():
    """Create and setup HuggingFace space."""
    try:
        install_git_lfs()
        
        # Verify token and get username
        username, token = verify_huggingface_token()
        space_name = "imagenet-classifier"
        repo_url = f"https://huggingface.co/spaces/{username}/{space_name}"
        
        # Set up git credentials
        subprocess.run(["git", "config", "--global", "credential.helper", "store"], check=True)
        
        # Store credentials
        git_credentials = f"https://{username}:{token}@huggingface.co\n"
        credentials_path = Path.home() / ".git-credentials"
        credentials_path.write_text(git_credentials)
        
        # Try to create space, but don't fail if it exists
        try:
            api = HfApi(token=token)
            create_repo(
                repo_id=f"{username}/{space_name}",
                token=token,
                repo_type="space",
                space_sdk="gradio",
                private=False
            )
            print(f"Created new HuggingFace space: {repo_url}")
        except Exception as e:
            if "already created" in str(e) or "already exists" in str(e):
                print(f"Using existing space: {repo_url}")
            else:
                raise e
        
        # Clone or pull the space
        if Path("space").exists():
            print("Updating existing space...")
            subprocess.run(["git", "-C", "space", "pull"], check=True)
        else:
            print(f"Cloning space from {repo_url}")
            subprocess.run(["git", "clone", repo_url, "space"], check=True)
        
        # Create or update space metadata
        metadata = {
            "title": "ImageNet Classifier",
            "emoji": "🖼️",
            "colorFrom": "blue",
            "colorTo": "red",
            "sdk": "gradio",
            "sdk_version": "4.7.1",
            "app_file": "app.py",
            "pinned": False
        }
        
        # Write metadata to README.md
        readme_content = "---\n"
        for key, value in metadata.items():
            readme_content += f"{key}: {value}\n"
        readme_content += "---\n\n"
        
        # Add the rest of README content
        readme_content += """# ImageNet Classifier

This is a ResNet50 model trained on ImageNet, achieving:
- Top-1 Accuracy: 79.26%
- Top-5 Accuracy: 94.51%

## Usage
1. Upload an image
2. Get predictions for the top 5 classes

## Model Details
- Architecture: ResNet50
- Training Dataset: ImageNet
- Input Size: 224x224
- Number of Classes: 1000
"""
        
        # Write README with metadata
        with open("space/README.md", "w") as f:
            f.write(readme_content)
        
        # Copy files to space directory
        files_to_copy = [
            "app.py",
            "requirements.txt",
            "model.ckpt",
            "imagenet_classes.json"
        ]
        
        for file in files_to_copy:
            if Path(file).exists():
                shutil.copy2(file, "space/")
                print(f"Copied {file} to space directory")
            else:
                print(f"Warning: {file} not found")
        
        # Copy examples directory
        if Path("examples").exists():
            shutil.copytree("examples", "space/examples", dirs_exist_ok=True)
            print("Copied examples directory")
        
        # Initialize git-lfs in the space directory
        subprocess.run(["git", "-C", "space", "lfs", "track", "*.ckpt"], check=True)
        subprocess.run(["git", "-C", "space", "lfs", "track", "*.bin"], check=True)
        subprocess.run(["git", "-C", "space", "lfs", "track", "*.pt"], check=True)
        
        # Push changes
        commands = [
            ["git", "-C", "space", "add", "."],
            ["git", "-C", "space", "commit", "-m", "Update deployment files"],
            ["git", "-C", "space", "push"]
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                if "nothing to commit" in str(e):
                    print("No changes to push")
                else:
                    raise e
        
        print(f"\nDeployment complete! Visit your space at:\n{repo_url}")
        
    except Exception as e:
        print(f"Error in create_huggingface_space: {str(e)}")
        raise

def ensure_env_file():
    """Create .env file if it doesn't exist."""
    env_file = Path('.env')
    if not env_file.exists():
        username = input("Enter your HuggingFace username: ").strip()
        with open(env_file, 'w') as f:
            f.write(f'HUGGINGFACE_USERNAME={username}\n')
        print("Created .env file with HuggingFace username")
    return True

def main():
    # Ensure we're in the deployment directory
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    print(f"Working directory: {os.getcwd()}")
    print("Starting deployment preparation...")
    
    try:
        ensure_env_file()
        # Verify HuggingFace token first
        verify_huggingface_token()
        
        download_imagenet_classes()
        download_example_images()
        copy_latest_checkpoint()
        create_huggingface_space()
        print("Deployment preparation complete!")
    except Exception as e:
        print(f"Error during deployment preparation: {e}")
        raise

if __name__ == "__main__":
    main() 