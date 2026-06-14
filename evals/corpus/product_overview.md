# Aether Cloud Storage - Product Overview

Aether Cloud Storage is a managed object storage service for storing,
backing up, and sharing files of any size. It is designed for teams that
need durable, encrypted storage with simple APIs.

## Key Features

- **Durability**: Every object is replicated across at least three
  availability zones, giving an annual durability target of 99.999999999%
  (11 nines).
- **Encryption**: All objects are encrypted at rest using AES-256, and all
  network traffic is encrypted in transit using TLS 1.2 or higher.
- **Versioning**: Buckets can enable versioning, which keeps every previous
  version of an object for 30 days by default before permanent deletion.
- **Access control**: Access is managed through Identity and Access
  Management (IAM) policies that can be scoped to individual buckets,
  folders, or objects.

## Storage Classes

Aether Cloud Storage offers three storage classes:

1. **Standard** - for frequently accessed data, with the lowest latency.
2. **Infrequent Access** - for data accessed less than once a month, at a
   reduced storage price but with a per-GB retrieval fee.
3. **Archive** - for long-term retention, with retrieval times of up to
   12 hours.

## Regions

Aether Cloud Storage is available in three regions: `us-east`, `eu-west`,
and `ap-southeast`. Data does not leave its selected region unless the
customer explicitly configures cross-region replication.
