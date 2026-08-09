# Reassembling the chunked archives

Each archive was split into <=10MB parts named `<name>.tar.gz.part000`,
`part001`, etc. The archive members are paths relative to `nets/`, so an
archive must be extracted into `nets/`:

```bash
cat <name>.tar.gz.part* > <name>.tar.gz
sha256sum <name>.tar.gz   # compare against checksums_full.sha256
mkdir -p ../nets && tar xzf <name>.tar.gz -C ../nets
```

`../reassemble.sh` does all of this for you, including the checksum check.

Archives: LTM_1000_ws, LTM_1000_mhk, LTM_1000_ke, LFC_240_ws, LFC_240_mhk,
LFC_240_ke, LFC_1000_mhk, LFC_5000_mhk, real_world.

- `checksums_full.sha256` - sha256 of each whole (reassembled) tar.gz.
- `checksums_parts.sha256` - sha256 of each individual chunk, for
  verifying transfer integrity of the parts themselves.
