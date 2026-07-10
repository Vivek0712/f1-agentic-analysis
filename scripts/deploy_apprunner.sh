#!/usr/bin/env bash
# Deploy the dashboard server to AWS App Runner.
#
#   AWS_PROFILE=<profile> ./scripts/deploy_apprunner.sh
#
# Creates (idempotently): an ECR repo, an instance role scoped to the outputs
# bucket + bedrock:InvokeModel, an ECR access role, and the App Runner service.
# Re-running pushes a new image and App Runner auto-deploys it.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO="f1-agentic-dashboard"
SERVICE="f1-agentic-dashboard"
BUCKET="${F1_OUTPUTS_BUCKET:-f1-agentic-analysis-outputs-$ACCOUNT}"
MODEL="${F1_BEDROCK_MODEL_ID:-us.amazon.nova-pro-v1:0}"
IMAGE="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:latest"

echo "== ECR repo"
aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION" >/dev/null
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

echo "== build + push (linux/amd64)"
docker build --platform linux/amd64 -t "$IMAGE" .
docker push "$IMAGE"

echo "== instance role (S3 bucket + Bedrock invoke)"
INSTANCE_ROLE="f1-dashboard-instance-role"
if ! aws iam get-role --role-name "$INSTANCE_ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$INSTANCE_ROLE" --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow",
      "Principal": {"Service": "tasks.apprunner.amazonaws.com"},
      "Action": "sts:AssumeRole"}]}' >/dev/null
fi
aws iam put-role-policy --role-name "$INSTANCE_ROLE" --policy-name f1-dashboard-access \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {\"Effect\": \"Allow\",
       \"Action\": [\"s3:GetObject\", \"s3:PutObject\", \"s3:ListBucket\", \"s3:CreateBucket\", \"s3:HeadBucket\"],
       \"Resource\": [\"arn:aws:s3:::$BUCKET\", \"arn:aws:s3:::$BUCKET/*\"]},
      {\"Effect\": \"Allow\",
       \"Action\": [\"bedrock:InvokeModel\", \"bedrock:InvokeModelWithResponseStream\"],
       \"Resource\": \"*\"}
    ]}"

echo "== ECR access role"
ECR_ROLE="f1-dashboard-ecr-access-role"
if ! aws iam get-role --role-name "$ECR_ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ECR_ROLE" --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow",
      "Principal": {"Service": "build.apprunner.amazonaws.com"},
      "Action": "sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ECR_ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
  sleep 10  # IAM propagation
fi

SOURCE_CONF="{
  \"ImageRepository\": {
    \"ImageIdentifier\": \"$IMAGE\",
    \"ImageRepositoryType\": \"ECR\",
    \"ImageConfiguration\": {
      \"Port\": \"8080\",
      \"RuntimeEnvironmentVariables\": {
        \"F1_OUTPUTS_BUCKET\": \"$BUCKET\",
        \"F1_BEDROCK_MODEL_ID\": \"$MODEL\"
      }
    }
  },
  \"AutoDeploymentsEnabled\": true,
  \"AuthenticationConfiguration\": {
    \"AccessRoleArn\": \"arn:aws:iam::$ACCOUNT:role/$ECR_ROLE\"
  }
}"

ARN=$(aws apprunner list-services --region "$REGION" \
      --query "ServiceSummaryList[?ServiceName=='$SERVICE'].ServiceArn | [0]" --output text)
if [ "$ARN" = "None" ] || [ -z "$ARN" ]; then
  echo "== create App Runner service"
  aws apprunner create-service --region "$REGION" \
    --service-name "$SERVICE" \
    --source-configuration "$SOURCE_CONF" \
    --instance-configuration "{\"Cpu\": \"1024\", \"Memory\": \"2048\",
      \"InstanceRoleArn\": \"arn:aws:iam::$ACCOUNT:role/$INSTANCE_ROLE\"}" \
    --health-check-configuration '{"Protocol": "HTTP", "Path": "/api/seasons",
      "Interval": 10, "Timeout": 5, "HealthyThreshold": 1, "UnhealthyThreshold": 5}' \
    --query "Service.{arn:ServiceArn,url:ServiceUrl,status:Status}" --output table
else
  echo "== service exists; new image auto-deploys (AutoDeploymentsEnabled)"
  aws apprunner describe-service --region "$REGION" --service-arn "$ARN" \
    --query "Service.{url:ServiceUrl,status:Status}" --output table
fi
