# Deploying the hosted demo

Everything in this file is **not yet done** - the code is ready and tested, the
accounts and servers are not created. Do the steps in order; each one ends
with a check so you know it worked before moving on.

```
Laptop A / B (Chrome) --https--> Azure Container App (this repo's Docker image)
                                        |
                                        +--https, user's own token--> Supabase (login + database)
```

The camera stays in each person's browser; only small JPEG snapshots reach the
server, and only numbers (scores, timestamps) are stored - never video.

---------------------------------------------------------------------------
## 0. Before anything: prove the image works on your laptop

```powershell
cd <path to your stress_project folder>
docker build -t stress-app .
# Local mode (no login, SQLite) - fastest sanity check of the container itself:
docker run --rm -p 8501:8501 --cpus=2 -m 4g -e STRESS_STORAGE_BACKEND=sqlite stress-app
```
Open http://localhost:8501 in Chrome (localhost counts as secure, so the camera
works). Calibrate -> Start -> Stop. If that works the image is good.

If `docker build` stops with *"looks like a Git LFS pointer"*, run `git lfs pull`
first - the two CNN weight files weren't downloaded.

Optional capacity check for the CPU you plan to rent (2 users at once):
```powershell
docker run --rm --cpus=2 -m 4g -v ${PWD}\scripts:/app/scripts -v ${PWD}\datasets:/app/datasets stress-app `
    python ../scripts/load_test.py --video ../datasets/UBFC-Phys/s11/vid_s11_T1.avi --users 2
```
Wants >= 8 fps per user. NOTE: this measures the models only. The real live app (browser
path + Streamlit) is ~2.2x slower: measured with ONE user it needs ~1.0-1.3 cores (8.2 fps on 2 CPUs,
~11 fps on 4). So **rent 4 vCPU for the demo** (the student cap) and/or set `POSE_EVERY_N_FRAMES = 3`
in `code/config.py` (see "Tuning" below). Details: `models/capacity_measurements.txt`.

---------------------------------------------------------------------------
## 1. Supabase (login + database) - free, no card

1. https://supabase.com -> sign in with GitHub -> **New project**. Pick the
   region nearest the Azure region you will use. Save the database password.
2. **SQL Editor -> New query**, paste all of `supabase/schema.sql`, **Run**.
   You should see "Success" and 4 tables under *Table Editor*, each marked
   with the RLS shield.
3. **Authentication -> Sign In / Providers -> Email**: set **"Allow new users
   to sign up" = ON** and **"Confirm email" = OFF**. This is not optional for the
   demo: with confirmation on, every sign-up sends mail through Supabase's built-in
   mailer, which allows only ~2-3 messages per hour on the free plan - you will hit
   HTTP 429 ("Too many attempts") and neither laptop will be able to register.
4. **Project Settings -> API Keys**: copy the **Project URL** and the **publishable**
   key (older projects call it **anon**). The app accepts either name:
   `SUPABASE_PUBLISHABLE_KEY` (preferred) or `SUPABASE_ANON_KEY` (legacy).
   The newer `sb_publishable_...` format is confirmed working with the pinned
   supabase-py.
   > NEVER copy the `service_role` / secret key into the app, Azure, GitHub or
   > this chat. It bypasses all privacy rules. The app refuses to start with it.
5. Run the app locally against it (still on your laptop):
   ```powershell
   docker run --rm -p 8501:8501 --cpus=2 -m 4g `
       -e STRESS_STORAGE_BACKEND=supabase `
       -e SUPABASE_URL=https://xxxx.supabase.co `
       -e SUPABASE_PUBLISHABLE_KEY=<publishable key> stress-app
   ```
   To run it from source instead of the image, put the same three names in
   `code/.streamlit/secrets.toml` (git-ignored) and just `streamlit run app.py` -
   see `.env.example`. Note `STRESS_STORAGE_BACKEND`: without it the app silently
   stays on local SQLite and never contacts Supabase.
   Create an account on the login page, calibrate, record, Stop. Reload the
   page and sign in again: your past session should be listed.
6. **Prove isolation** with two THROWAWAY accounts (not your demo accounts - the
   script temporarily rewrites account A's calibration baseline and restores it):
   ```powershell
   python scripts\rls_check.py --url https://xxxx.supabase.co --key <publishable key> `
       --email-a a@x.com --password-a ... --email-b b@x.com --password-b ...
   ```
   It must end with `ALL CHECKS PASSED`. Show this to the teacher.

Free projects **pause after 7 idle days**. Create the project close to the demo,
and use `deploy/github-workflow-keepalive.yml` (below) to ping it daily.

---------------------------------------------------------------------------
## 2. Azure account (students, no card)

Sign up at https://azure.microsoft.com/en-us/free/students with your college
email. **Never press "Upgrade to Pay-As-You-Go"** - it permanently removes the
student credit. Then, once:

```powershell
winget install -e --id Microsoft.AzureCLI      # restart the terminal afterwards
az login
az account show                                 # should say "Azure for Students"
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az provider register --namespace Microsoft.ContainerRegistry --wait
```
Student subscriptions only allow some regions. Find yours: portal ->
*Resource groups -> Create* shows the allowed regions in the dropdown. Use one
of those for `$LOC` below (and a Supabase region near it).

---------------------------------------------------------------------------
## 3. Put the image on Azure

The image is ~3 GB (TensorFlow + MediaPipe + two CNNs). `docker push` uploads all
of it the first time (later pushes only send the changed layers), so do this step on a
fast home connection, not the college network.

```powershell
$RG  = "stress-rg"
$LOC = "<an allowed region, e.g. uaenorth>"
$ACR = "stressacr$(Get-Random -Maximum 99999)"       # must be globally unique, lowercase
$ENV = "stress-env"
$APP = "stress-app"

az group create -n $RG -l $LOC
az acr create -n $ACR -g $RG --sku Basic --admin-enabled true
az acr login -n $ACR
docker build -t "$ACR.azurecr.io/stress-app:v1" .
docker push  "$ACR.azurecr.io/stress-app:v1"

az containerapp env create -n $ENV -g $RG -l $LOC
$USER = az acr credential show -n $ACR --query username -o tsv
$PASS = az acr credential show -n $ACR --query "passwords[0].value" -o tsv

az containerapp create -n $APP -g $RG --environment $ENV `
    --image "$ACR.azurecr.io/stress-app:v1" `
    --registry-server "$ACR.azurecr.io" --registry-username $USER --registry-password $PASS `
    --target-port 8501 --ingress external `
    --cpu 2 --memory 4Gi --min-replicas 1 --max-replicas 1 `
    --secrets "supabase-url=https://xxxx.supabase.co" "supabase-key=<publishable key>" "signup-code=<an invite code you choose>" `
    --env-vars "STRESS_STORAGE_BACKEND=supabase" "SUPABASE_URL=secretref:supabase-url" "SUPABASE_PUBLISHABLE_KEY=secretref:supabase-key" "STRESS_SIGNUP_CODE=secretref:signup-code"

az containerapp show -n $APP -g $RG --query properties.configuration.ingress.fqdn -o tsv
```
The last command prints your public address (`https://<that value>`), e.g.
`stress-app.<random>.<region>.azurecontainerapps.io`. No domain needs buying.

Keep **min = max = 1 replica**: each person's live session lives in the memory
of the one running copy, so a second copy would drop sessions.

`STRESS_SIGNUP_CODE` makes the "Create account" form ask for that invite code, so
strangers who find the public address cannot sign up (give it to your friend and the
teacher). The app also refuses more than 6 simultaneous sessions
(`STRESS_MAX_SESSIONS`) so the demo server can't be swamped.

Check: open the address -> login page appears -> sign in -> Calibrate asks for
camera permission (it works because the address is https).

Useful commands
```powershell
az containerapp logs show -n $APP -g $RG --follow                 # live logs
az containerapp update -n $APP -g $RG --cpu 4 --memory 8Gi        # demo day: more CPU
az containerapp update -n $APP -g $RG --min-replicas 0            # idle between sessions: costs ~$0 (cold start ~1 min)
az containerapp update -n $APP -g $RG --min-replicas 1            # warm it up again
```
Student credit is $100; 2 vCPU / 4 GiB running 24/7 is only a few dollars a
day, and scaling to zero costs nothing - so scale to zero when not testing and
set `--min-replicas 1` + 4 vCPU the day before the demo.

If a step is refused with a policy/quota/region error, tell me the exact error
text - the fallbacks are an Azure VM (same credit), Hugging Face Spaces, or
Railway.

---------------------------------------------------------------------------
## 4. Auto-redeploy from Git

Template: `deploy/github-workflow-azure.yml`. Copy it to
`.github/workflows/deploy.yml`, then in GitHub -> repo -> *Settings -> Secrets
and variables -> Actions* add:

| Secret | Value |
|---|---|
| `AZURE_CREDENTIALS` | output of `az ad sp create-for-rbac --name stress-deploy --role contributor --scopes /subscriptions/<id>/resourceGroups/stress-rg --sdk-auth` |
| `ACR_NAME` | the `$ACR` name from step 3 |

After that, every push to `main` rebuilds the image and updates the live app.
(The models are Git-LFS files: each build downloads ~51 MB of LFS, and GitHub's
free LFS bandwidth is 1 GB/month - about 20 deploys. Don't push to `main` in a
loop.) If creating the service principal is blocked on a student account, run
the three `docker`/`az` update commands from step 3 by hand instead
(`docker build` -> `docker push` -> `az containerapp update --image ...`).

Supabase keep-alive: copy `deploy/github-workflow-keepalive.yml` to
`.github/workflows/` and add secrets `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`.

---------------------------------------------------------------------------
## 5. Demo-day checklist

- [ ] Day before: `--min-replicas 1`, `--cpu 4 --memory 8Gi`; Supabase project awake
      (open the dashboard once); run `scripts/rls_check.py` one last time.
- [ ] Both accounts created, each calibrated **once** beforehand (their baselines
      are saved to their account, so on the day you go straight to Start).
- [ ] On each laptop: Chrome, **Ethernet** or phone hotspot as backup, camera
      permission already granted for the address (visit once beforehand).
- [ ] Don't reload the page mid-demo: it signs you out (by design).
- [ ] Emergency fallback if Azure or the venue network misbehaves - run the app on
      your own laptop and expose it with a free tunnel (camera still works because
      the tunnel is https):
      ```powershell
      winget install -e --id Cloudflare.cloudflared
      cd code; $env:STRESS_STORAGE_BACKEND="sqlite"; python -m streamlit run app.py
      cloudflared tunnel --url http://localhost:8501      # prints https://<random>.trycloudflare.com
      ```
      (SQLite mode has no login: use it only as a last resort.)

---------------------------------------------------------------------------
## Tuning and troubleshooting

- **Stress readout updates slowly / "N fps sent . X s round trip" in the video's corner is
  high**: the server is CPU-bound. Raise CPU (`--cpu 4`), and/or run Pose every 3rd frame:
  set `POSE_EVERY_N_FRAMES = 3` in `code/config.py` **only after**
  `python experiment_mediapipe_tasks_parity.py --mode pose-cadence --pose-every 3`
  shows the posture signals still agree (see the result recorded in
  `models/pose_cadence_parity.txt`).
- **Camera blocked**: the browser shows a message on the video. Click the camera icon in
  the address bar -> Allow -> "Try again". Camera access requires https (or localhost).
- **Page loads but stays on "Please wait" / WebSocket errors behind a proxy**: add
  `--server.enableCORS=false --server.enableXsrfProtection=false` to the `CMD` in the
  Dockerfile (and rebuild). Only needed if the first deploy shows connection errors.
- **Everyone is logged out after a deploy**: expected - sessions live in the container's memory.
- **"HR BPM" shows `--`**: the heart-rate estimate needs >= 12 processed frames/second;
  it carries zero weight in the stress score either way (`W_RPPG = 0`).

---------------------------------------------------------------------------
## Verified against a real Supabase project (2026-10-11)

Done on project `dctm****.supabase.co`, before any Azure deployment:

| Check | Result |
|---|---|
| `sb_publishable_...` key accepted by the pinned supabase-py | yes |
| Anonymous caller blocked on all 4 tables | yes - `permission denied` (the `revoke ... from anon` bites before RLS) |
| `scripts/rls_check.py` - A vs B isolation, 18 checks | ALL PASSED |
| Login / logout / session create / read / update | passed |
| Data still present after logout + fresh login | passed |
| Delete cascades session -> samples | passed |
| App renders hosted mode, Past sessions read from the live DB | passed |
| SQLite fallback still works, local DBs byte-identical | passed |

Not yet done: Azure deployment, the two-laptop rehearsal, and a real-webcam run
against the hosted URL.
