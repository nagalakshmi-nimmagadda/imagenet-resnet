#!/bin/bash
set -e

INSTANCE_TYPE="g6.12xlarge"  # 4 NVIDIA A10G GPUs
SPOT_PRICE="2.50"           # Maximum spot price (adjust based on market)
VOLUME_SIZE="200"           # GB for root volume
AMI_ID="ami-053b12d3152c0cc71"  # Ubuntu 22.04 Deep Learning AMI

# Add instance configuration
SUBNET_ID="subnet-01afc72f0c4c64393"  # Your subnet ID
SECURITY_GROUP="sg-0a80002bf2156b66b"  # Your security group

launch_spot() {
    aws ec2 request-spot-instances \
        --instance-count 1 \
        --launch-specification "{
            \"ImageId\": \"${AMI_ID}\",
            \"InstanceType\": \"${INSTANCE_TYPE}\",
            \"SubnetId\": \"${SUBNET_ID}\",
            \"SecurityGroupIds\": [\"${SECURITY_GROUP}\"],
            \"BlockDeviceMappings\": [{
                \"DeviceName\": \"/dev/sda1\",
                \"Ebs\": {
                    \"VolumeSize\": ${VOLUME_SIZE},
                    \"VolumeType\": \"gp3\",
                    \"Iops\": 16000,
                    \"Throughput\": 1000
                }
            }],
            \"EbsOptimized\": true
        }" \
        --spot-price "$SPOT_PRICE" \
        --instance-interruption-behavior stop

    echo "Waiting for spot instance to be ready..."
    aws ec2 wait instance-running
}

# Add spot instance termination check
check_termination() {
    while true; do
        if curl -s http://169.254.169.254/latest/meta-data/spot/termination-time | grep -q .*T.*Z; then
            echo "Spot instance termination notice received. Saving checkpoint..."
            pkill -15 -f "train.py"  # Send SIGTERM to training process
            sleep 30
            echo "Shutting down..."
            sudo shutdown -h now
        fi
        sleep 5
    done
}

# Add spot price monitoring
monitor_spot_price() {
    local region=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
    aws ec2 describe-spot-price-history \
        --instance-types ${INSTANCE_TYPE} \
        --product-descriptions "Linux/UNIX" \
        --start-time $(date -u +"%Y-%m-%dT%H:%M:%SZ") \
        --region ${region} \
        --query 'SpotPriceHistory[*].[AvailabilityZone,SpotPrice]' \
        --output table
}

# Add function to check GPU health
check_gpu_health() {
    nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader
}

case "$1" in
    "launch")
        launch_spot
        ;;
    "check-termination")
        check_termination
        ;;
    "monitor-price")
        monitor_spot_price
        ;;
    "check-gpu")
        check_gpu_health
        ;;
    *)
        echo "Usage: $0 {launch|check-termination|monitor-price|check-gpu}"
        exit 1
        ;;
esac 