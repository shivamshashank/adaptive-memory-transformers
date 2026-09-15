# Security Policy

We take the security of **Adaptive KV-Cache Compression** seriously. Although
this is a research repository and not a deployed service, vulnerabilities may
affect code execution, dependency integrity, experiment isolation, credentials,
or the handling of private datasets and model access tokens.

## Supported Versions

This project has not released a versioned software package. Security fixes are
currently applied to the default branch:

| Version | Supported |
|---|---|
| `main` | Yes |
| Older revisions | Best effort |

Use pinned environments and review model, dataset, and dependency licenses
before running experiments. Never commit API keys, Hugging Face tokens,
private prompts, private datasets, or raw outputs containing sensitive data.

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security-related bugs.**

To report a vulnerability:

1. **Email:** Send a detailed report to
   **<shivamshashank@users.noreply.github.com>**.
2. **GitHub Security Advisory:** If enabled for the repository, draft a private
   advisory through GitHub's Security tab.

### What to Include

- A description of the vulnerability and potential impact.
- Steps to reproduce, including commands, configurations, or environment
  details.
- Whether credentials, private data, model artifacts, or generated outputs are
  exposed.
- Suggested mitigation or patch, if available.

## Response Process

1. **Acknowledgment:** We aim to acknowledge receipt within 48 hours.
2. **Evaluation:** We will investigate the issue and determine its severity.
3. **Fix and disclosure:** We will prepare a fix or mitigation and coordinate a
   public disclosure where appropriate.

We thank you for helping keep the research code, experiment environments, and
shared results secure.
