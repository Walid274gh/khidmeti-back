# MongoDB Encryption-at-Rest (AES-256) — the base layer

This is the **whole-database** AES-256 encryption. It is transparent: every field
stays fully searchable (regex, range, sort) because decryption happens in the
storage engine, below the query layer. Zero application code, zero query changes.

The application-level field encryption of `email`/`phoneNumber` (see
`src/common/crypto/`) sits **on top** of this as defense-in-depth — it survives
even a raw file/backup theft where the storage key leaked.

## Option A — MongoDB Atlas (managed)

Encryption-at-rest is **on by default** with AES-256. Nothing to configure.
For a customer-managed key, enable Atlas "Encryption at Rest using your Key
Management" (AWS KMS / Azure Key Vault / GCP KMS) in the project settings.

## Option B — self-hosted (WiredTiger, MongoDB Enterprise)

WiredTiger native encryption requires **MongoDB Enterprise**. Use a KMIP server
in production; a local keyfile is acceptable for a single self-hosted node.

```yaml
# mongod.conf
security:
  enableEncryption: true
  encryptionCipherMode: AES256-GCM
  encryptionKeyFile: /etc/mongodb/encryption-keyfile   # 32-byte base64 key, chmod 600
```

Generate the keyfile once and protect it:

```bash
openssl rand -base64 32 > /etc/mongodb/encryption-keyfile
chmod 600 /etc/mongodb/encryption-keyfile
chown mongodb:mongodb /etc/mongodb/encryption-keyfile
```

Encryption applies to **newly written data** — initialise it on an empty dbpath,
or dump → enable → restore to encrypt an existing dataset.

## Option C — self-hosted Community edition (no WiredTiger encryption)

WiredTiger encryption is Enterprise-only. On Community, encrypt at the layer
below MongoDB instead: **LUKS full-disk encryption** on the volume holding
`dbpath` (`cryptsetup luksFormat`). Same AES-256-at-rest guarantee, provided by
the OS, transparent to MongoDB and to queries.

For the docker-compose setup: run the `mongo` volume on a LUKS-encrypted host
directory, or use an encrypted cloud block volume (EBS/PD encryption is AES-256
and on-by-default on most providers).

---

**Bottom line:** Atlas or an encrypted volume gives you "AES-256 for the whole
database, search still works" with no code. The field-level layer in this
codebase is the extra hardening for the two PII fields that matter most.
