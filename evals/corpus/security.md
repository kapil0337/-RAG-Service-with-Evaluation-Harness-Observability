# Aether Cloud Storage - Security and Compliance

## Data Encryption

All objects stored in Aether Cloud Storage are encrypted at rest using
AES-256 with keys managed by Aether's key management service (KMS).
Customers may optionally provide their own encryption keys
("customer-managed keys") for additional control.

## Access Logging

Every read and write request can optionally be logged to an audit bucket.
Audit logs include the requester identity, timestamp, source IP address,
and the action performed. Audit logs are retained for 400 days.

## Compliance Certifications

Aether Cloud Storage has been independently audited and holds the
following certifications:

- SOC 2 Type II
- ISO 27001
- HIPAA eligibility (when used with a signed Business Associate Agreement)

## Multi-Factor Authentication

Accounts with billing or IAM administrator roles are required to enable
multi-factor authentication (MFA). Accounts without MFA enabled after a
30-day grace period will have administrator access suspended until MFA is
configured.

## Incident Response

Security incidents affecting customer data are reported to affected
customers within 72 hours of confirmation, in line with the incident
response policy published in the Aether Trust Center.
