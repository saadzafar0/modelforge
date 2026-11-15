# ModelForge DVC remote setup

The project's default DVC remote is configured as follows:

| Setting | Value |
| --- | --- |
| DagsHub repository | https://dagshub.com/saadzafar0/modelforge |
| DVC remote name | `origin` |
| DVC storage URL | `s3://dvc` |
| S3-compatible endpoint | `https://dagshub.com/saadzafar0/modelforge.s3` |

The Git remote named `origin` still points to GitHub. Git and DVC have separate remote configurations, so using the same name is valid.

## What is configured

- `.dvc/config` contains the shared endpoint and default remote, with no credentials.
- `.dvc/config.local` contains the local S3 credentials and is ignored by Git.
- DVC's local state cache is configured in `.dvc/tmp/site-cache` on this machine to avoid a permission error at the default system cache location. This setting is local and is not shared through Git.
- `dvc[s3]` is installed in `.venv-ml` for the pipeline.

S3-compatible authentication uses the DagsHub token as both `access_key_id` and `secret_access_key`. The HTTP settings `auth basic`, `user`, and `password` are not used for this remote. No AWS account or AWS credentials are needed.

## Replace the exposed token

The token shared in the conversation was used to verify the connection. Revoke that token in DagsHub's user settings and generate a replacement. Enter the replacement only in your local terminal using the hidden prompt below; do not paste it into chat or commit it to Git.

Run from the repository root in Bash:

```bash
source .venv-ml/bin/activate
read -rsp "New DagsHub token: " MODELFORGE_DVC_TOKEN
printf '\n'
dvc remote modify --local origin access_key_id "$MODELFORGE_DVC_TOKEN"
dvc remote modify --local origin secret_access_key "$MODELFORGE_DVC_TOKEN"
unset MODELFORGE_DVC_TOKEN
```

## Verification completed

On 2026-11-15, an isolated temporary DVC project successfully:

1. Tracked one unique connection-check file.
2. Uploaded the file using `dvc push` and the configured remote.
3. Removed the temporary project's local file and cache.
4. Downloaded the file using `dvc pull`.
5. Verified that the downloaded bytes exactly matched the uploaded bytes.
6. Deleted only the unique remote connection-check object.

This verifies DVC upload and download access. It does not stand in for versioning the actual project dataset, running the ML pipeline, or creating MLflow experiments.

The command transcript and screenshots are in [`screenshots/deleiverable_1`](../../screenshots/deleiverable_1/).

## Add the real dataset later

After choosing and ingesting the dataset, for example at `data/raw/dataset.csv`:

```bash
dvc add data/raw/dataset.csv
dvc push
git add data/raw/dataset.csv.dvc data/raw/.gitignore
git commit -m "Version initial classification dataset"
git push
```

For a new team checkout, install `dvc[s3]`, configure the teammate's own local credentials using the hidden-prompt commands above, and run `dvc pull`. Private repositories require appropriate DagsHub access.

When the pipeline exists, commit `dvc.yaml` and `dvc.lock` to Git and push their DVC-tracked outputs using `dvc push`.

## References

- [DagsHub's current DVC remote instructions](https://dagshub.com/docs/feature_guide/dagshub_storage/#working-with-the-dvc-remote)
- [DVC getting started](https://doc.dvc.org/start?tab=Mac-Linux)
