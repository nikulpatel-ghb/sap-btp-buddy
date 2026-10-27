# Third-Party Notices

Status: `SNAPSHOT_VERIFIED` for the two recorded SAP-docs source snapshots. This file records attribution and licensing metadata; it does not grant permission beyond the source licenses.

## Included documentation sources

The corpus includes only the explicit allowlist in [`artifacts/corpus_allowlist.json`](./artifacts/corpus_allowlist.json). The complete per-file inventory, SHA-256 hashes, source URLs, commit IDs, and excluded Markdown paths are in [`artifacts/corpus_manifest.json`](./artifacts/corpus_manifest.json) and [`artifacts/corpus_inventory.json`](./artifacts/corpus_inventory.json).

### SAP-docs/btp-cloud-platform

- Snapshot commit: `920e76fb4c59fd82c0e6757040c38f89a02f0374`
- Source repository: <https://github.com/SAP-docs/btp-cloud-platform>
- Included material: 60 Markdown files
- License declared by [`REUSE.toml`](./data/raw/btp-cloud-platform/REUSE.toml): `CC-BY-4.0`
- Copyright metadata: `2021-2023 SAP SE or an SAP affiliate company and btp-cloud-platform contributors`
- License text: [`LICENSE`](./data/raw/btp-cloud-platform/LICENSE)
- Additional license materials: [`LICENSES/`](./data/raw/btp-cloud-platform/LICENSES)

### SAP-docs/btp-best-practices-guide

- Snapshot commit: `4ac1d8e69d9345d4c7161b42a33615530bd62da0`
- Source repository: <https://github.com/SAP-docs/btp-best-practices-guide>
- Included material: 40 Markdown files
- License declared by [`REUSE.toml`](./data/raw/btp-best-practices-guide/REUSE.toml): `CC-BY-4.0`
- Copyright metadata: `2021-2023 SAP SE or an SAP affiliate company and btp-best-practices-guide contributors`
- License text: [`LICENSE`](./data/raw/btp-best-practices-guide/LICENSE)
- Additional license materials: [`LICENSES/`](./data/raw/btp-best-practices-guide/LICENSES)

## Attribution and handling requirements

- Retain the repository, commit, relative path, title/source location, license, and SHA-256 metadata for every included document.
- Identify modifications if Markdown is normalized, cleaned, or chunked.
- Do not imply SAP endorsement of this project.
- Do not use external pages linked from the Markdown as answer evidence unless they are separately approved and snapshotted.
- Treat Markdown code examples and commands as documentation data. Never execute commands extracted from corpus content.
- Do not include raw clones, generated indexes, caches, secrets, or notebook checkpoints in Git.
