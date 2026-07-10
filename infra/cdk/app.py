"""Minimal CDK sketch for the batch path: S3 + IAM for Bedrock batch inference,
EventBridge -> Fargate for the deterministic layer. Slow-loop AgentCore wiring
is account-specific and documented in ../architecture.md.
Deploy: cdk deploy (after cdk bootstrap; requires aws-cdk-lib).
"""
import aws_cdk as cdk
from aws_cdk import aws_s3 as s3, aws_iam as iam

app = cdk.App()
stack = cdk.Stack(app, "F1AgenticAnalysis")

lake = s3.Bucket(stack, "TelemetryLake",
                 versioned=True,
                 block_public_access=s3.BlockPublicAccess.BLOCK_ALL)

audit = s3.Bucket(stack, "AuditRecords",
                  object_lock_enabled=True,   # immutable audit trail
                  block_public_access=s3.BlockPublicAccess.BLOCK_ALL)

batch_role = iam.Role(stack, "BedrockBatchRole",
                      assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"))
lake.grant_read_write(batch_role)

cdk.CfnOutput(stack, "LakeBucket", value=lake.bucket_name)
cdk.CfnOutput(stack, "AuditBucket", value=audit.bucket_name)
app.synth()
