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

# This machine's Azure CLI may already be signed in as a DIFFERENT account.
az account show --query "{sub:name, user:user.name}" -o json
az logout                                       # only if the above is not your student account
az login                                        # sign in with the college email

az account show --query name -o tsv             # must say "Azure for Students"
az account list -o table                         # if several, pick the student one:
# az account set --subscription "Azure for Students"

az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az provider register --namespace Microsoft.ContainerRegistry --wait
```
Student subscriptions only allow some regions, and the list has been narrowed
over time. Find yours: portal -> *Resource groups -> Create* shows the allowed
regions in the dropdown. Use one of those for `$LOC` below (and a Supabase
region near it). Prefer `centralindia` if it is offered - it is the closest to
Bangalore, and the Container Apps Consumption rates are identical to the US
regions, so there is nothing to gain by going further away.

The same subscription type caps total vCPU quota (commonly 4). That is exactly
the demo configuration below, so there is no headroom for a second running app
- delete or scale down anything else first.

---------------------------------------------------------------------------
## 3. Put the image on Azure

### 3a. What already exists (verified read-only, 2026-10-15)

Everything in this table is already created and working - normally you only
need 3b. Section 3c is only for rebuilding from nothing.

| Resource | Name | Notes |
|---|---|---|
| Subscription | Azure for Students (`39643323-d40d-4cee-8be3-5f1d0d00a8b1`) | |
| Resource group | `stress-detection-demo-rg` | its own location is `southindia`; allowed because the subscription's region policy ("Allowed resource deployment regions", mode *Indexed*) does not apply to resource groups |
| Region of every resource | `eastasia` | the policy allows only `polandcentral`, `koreacentral`, `eastasia`, `indiasouthcentral`, `austriaeast`, and Container Apps is not offered in `indiasouthcentral` |
| Container registry | `stressstresstracker2026` (Basic) | admin user **off**, anonymous pull **off** |
| Log Analytics | `stress-logs` | 30-day retention; receives the app's console logs |
| Container Apps environment | `stress-env` | an **Express** environment, Consumption profile only - see the notes below |
| Pull identity | `stress-app-pull-id` (user-assigned managed identity) | its only role: `AcrPull` on the registry |
| Container app | `stress-app` | 4 vCPU / 8 GiB; min 0 / max 1 replicas; external HTTPS ingress on port 8501, transport HTTP/1.1, HTTP redirected to HTTPS, no IP restrictions; image pulled by digest through `stress-app-pull-id` |
| Public address | `https://stress-app.ambitiousplant-4f39aef6.eastasia.azurecontainerapps.io` | |

App settings (secret **names** only - the values exist only in Azure):

| Setting | Value |
|---|---|
| `STRESS_STORAGE_BACKEND` | `supabase` |
| `SUPABASE_URL` | secret reference `supabase-url` |
| `SUPABASE_PUBLISHABLE_KEY` | secret reference `supabase-key` - the publishable key only, never a secret / service-role key |
| `STREAMLIT_CLIENT_SHOW_ERROR_DETAILS` | `none` - visitors never see Python tracebacks |
| `STREAMLIT_CLIENT_TOOLBAR_MODE` | `viewer` - hides Streamlit's developer menu |
| `STRESS_SIGNUP_CODE` | secret reference `signup-code` - **obsolete**: the current code ignores it. Remove both at the next deployment (3b). |

As of 2026-10-15 the app runs image `stress-app@sha256:03582187…` (tag `v1`).
That image predates the email/password rework - it still shows the invite-code
form - so deploying the current code needs a new image (3b).

Express environment notes (each one was hit during the first deployment):

* **No system-assigned identity**: Express rejects it, so image pulls use the
  user-assigned `stress-app-pull-id`.
* **No session affinity** (sticky sessions): not needed, because max replicas is 1.
* `az containerapp stop` / `start` are not in Azure CLI 2.88, and
  `az containerapp revision deactivate` fails: use the ARM start/stop actions (3b).
* `az containerapp logs show` fails on Express apps (`KeyError:
  'eventStreamEndpoint'`): read the logs in Log Analytics instead (3b).
* Turn ingress on while the app is **running** (or at create time). Ingress
  enabled while the app was stopped returned 404 `{"error":"Not found"}` until it
  was toggled off and on with the app running.
* With ingress **off**, min replicas 0 did not scale the app to zero; the stop
  action does.
* `az acr build` (ACR Tasks) is blocked on this subscription
  (`TasksOperationsNotAllowed`): build the image locally and push it.

### 3b. Deploying new code to the existing app

The image is ~3 GB uncompressed (~0.9 GB to push, compressed), so push from a
fast connection, not the college network.

```powershell
$SUB   = "39643323-d40d-4cee-8be3-5f1d0d00a8b1"
$RG    = "stress-detection-demo-rg"
$ACR   = "stressstresstracker2026"
$APP   = "stress-app"
$APPID = "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.App/containerApps/$APP"
$TAG   = "v2"                                        # a new tag for every build

az account set --subscription $SUB
docker build --platform linux/amd64 -t "$ACR.azurecr.io/stress-app:$TAG" .
az acr login -n $ACR                                # your Entra sign-in - no registry password
docker push "$ACR.azurecr.io/stress-app:$TAG"

# Deploy by immutable digest, never by tag.
$DIGEST = az acr repository show -n $ACR --image "stress-app:$TAG" --query digest -o tsv
az containerapp update -n $APP -g $RG --image "$ACR.azurecr.io/stress-app@$DIGEST"

# Once: remove the obsolete invite code (the setting first, then the secret it uses).
az containerapp update -n $APP -g $RG --remove-env-vars STRESS_SIGNUP_CODE
az containerapp secret remove -n $APP -g $RG --secret-names signup-code
```

Start / stop - a stopped app runs no replicas and so uses no compute:

```powershell
az rest --method post --url "https://management.azure.com$APPID/start?api-version=2024-03-01"
az rest --method post --url "https://management.azure.com$APPID/stop?api-version=2024-03-01"
az containerapp show -n $APP -g $RG --query properties.runningStatus -o tsv     # Running / Stopped
az containerapp replica list -n $APP -g $RG --query "length(@)" -o tsv         # running replicas
```

Logs: portal -> `stress-logs` -> *Logs*, query
`ContainerAppConsoleLogs_CL | where ContainerAppName_s == "stress-app" | order by TimeGenerated desc`.

Replicas: keep **max 1** - each person's live session lives in the memory of the
one running copy, so a second copy would drop sessions. **Min 0** is the normal
setting. For the demo window set min 1, so Azure cannot scale the app down under a
live session (a Streamlit session runs over one long-lived WebSocket, while Azure's
HTTP scaler counts new requests):

```powershell
az containerapp update -n $APP -g $RG --min-replicas 1     # demo window
az containerapp update -n $APP -g $RG --min-replicas 0     # afterwards (and/or stop the app)
```

Sign-in is email + password only, and registration is open - there is no invite
code any more. Two Supabase dashboard settings decide how new accounts behave
(Authentication -> Sign In / Providers -> Email, and -> URL Configuration):
**Confirm email** (ON = people must click a link before they can sign in; OFF =
they are signed in immediately) and **Site URL** (where that link lands - set it to
the app's https address). The app handles both. The app also refuses more than 6
simultaneous sessions (`STRESS_MAX_SESSIONS`) so the demo server can't be swamped.

Check: open the address -> login page appears -> sign in -> Calibrate asks for
camera permission (it works because the address is https).

### 3c. Rebuilding from scratch (only if the resources in 3a are gone)

Not needed today - 3a already exists. Read 3a's Express notes first. Never paste
the Supabase values or the workspace key into a shared terminal history; load
them into variables from local, git-ignored files.

```powershell
$SUB  = "39643323-d40d-4cee-8be3-5f1d0d00a8b1"
$RG   = "stress-detection-demo-rg"
$LOC  = "eastasia"                                   # an allowed region that offers Container Apps
$ACR  = "<globally unique, lowercase>"
$ENV  = "stress-env"
$APP  = "stress-app"
$LOGS = "stress-logs"
$ID   = "stress-app-pull-id"

az account set --subscription $SUB
az group create -n $RG -l $LOC
az acr create -n $ACR -g $RG -l $LOC --sku Basic     # no --admin-enabled: no registry password
# build, log in and push the image as in 3b, then set $DIGEST

az monitor log-analytics workspace create -g $RG -n $LOGS -l $LOC
$WSID  = az monitor log-analytics workspace show -g $RG -n $LOGS --query customerId -o tsv
$WSKEY = az monitor log-analytics workspace get-shared-keys -g $RG -n $LOGS --query primarySharedKey -o tsv
az containerapp env create -n $ENV -g $RG -l $LOC `
    --logs-destination log-analytics --logs-workspace-id $WSID --logs-workspace-key $WSKEY
Remove-Variable WSKEY
# On this subscription (Azure CLI 2.88) that produced an *Express* environment.

az identity create -g $RG -n $ID -l $LOC
$UAMI      = az identity show -g $RG -n $ID --query id -o tsv
$PRINCIPAL = az identity show -g $RG -n $ID --query principalId -o tsv
$ACRID     = az acr show -n $ACR -g $RG --query id -o tsv
az role assignment create --assignee-object-id $PRINCIPAL --assignee-principal-type ServicePrincipal `
    --role AcrPull --scope $ACRID
# wait 1-2 minutes for the role assignment to take effect

az containerapp create -n $APP -g $RG --environment $ENV --workload-profile-name Consumption `
    --image "$ACR.azurecr.io/stress-app@$DIGEST" `
    --user-assigned $UAMI --registry-server "$ACR.azurecr.io" --registry-identity $UAMI `
    --cpu 4 --memory 8Gi --min-replicas 0 --max-replicas 1 `
    --ingress external --target-port 8501 --transport http `
    --secrets "supabase-url=$SUPABASE_URL" "supabase-key=$SUPABASE_PUBLISHABLE_KEY" `
    --env-vars "STRESS_STORAGE_BACKEND=supabase" "SUPABASE_URL=secretref:supabase-url" `
               "SUPABASE_PUBLISHABLE_KEY=secretref:supabase-key" `
               "STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=none" "STREAMLIT_CLIENT_TOOLBAR_MODE=viewer"

az containerapp show -n $APP -g $RG --query properties.configuration.ingress.fqdn -o tsv
```
Ingress at create time follows Microsoft's Express documentation; on this
subscription the app was created without ingress and it was added later. If the
address answers 404 `{"error":"Not found"}`, run `az containerapp ingress disable`
and then `az containerapp ingress enable --type external --target-port 8501
--transport http` while the app is running. The last command prints the public
address; no domain needs buying.

If a step is refused with a policy/quota/region error, tell me the exact error
text - the fallbacks are an Azure VM (same credit), Hugging Face Spaces, or
Railway.

### What it costs

Rates below are from Azure's public retail price API (13 Oct 2026, Consumption
plan - identical in `centralindia`, `eastus` and `eastasia`): vCPU active
$0.000024/vCPU-second, memory $0.000003/GiB-second, requests $0.40/million.
Free every calendar month, per subscription: **180,000 vCPU-seconds, 360,000
GiB-seconds, 2 million requests** (health-probe requests are not billable).

| Config | Running cost | Free grant covers |
|---|---|---|
| scaled to zero | **$0.00** | - |
| 2 vCPU / 4 GiB | $0.216 / hour | first **25 h** each month |
| 4 vCPU / 8 GiB | $0.432 / hour | first **12.5 h** each month |

Caveat: Azure Cost Management for October 2026 showed the app's vCPU and memory
usage billed (about INR 22 for a few hours of running) although the free grant
should have covered it - check the invoice / credit balance before relying on the
free hours.

So a realistic demo week - say 30 hours of 4 vCPU uptime across testing,
rehearsal and the demo itself - costs about **$7.60** after the free grant,
plus ACR Basic at $0.1666/day (~$5/month) and $0 for Log Analytics (the first
5 GB/month of ingestion is free and this app logs a few MB). Call it **under
$15 for the whole exercise** against the $100 credit.

What would blow the budget is leaving 4 vCPU running all month: 730 h is about
$310. **Scale to zero whenever you are not actively testing**, and only set
`--min-replicas 1` from the evening before the demo.

```powershell
az rest --method post --url "https://management.azure.com$APPID/stop?api-version=2024-03-01"   # done for today
```

---------------------------------------------------------------------------
## 4. Auto-redeploy from Git

Template: `deploy/github-workflow-azure.yml`. It deploys **only when you click
"Run workflow"**, pauses for an approval, and authenticates with **OIDC** -
there is no Azure password or client secret stored in GitHub at all. (The older
`az ad sp create-for-rbac --sdk-auth` recipe puts a long-lived client secret in
a repo secret; anyone with write access to the repo can then deploy, and the
secret stays valid until manually rotated. OIDC tokens are minted per run,
expire in minutes, and are only issued to the repo/branch/environment named in
the federated credential.)

**a. Create the identity Azure will trust** (portal: *Microsoft Entra ID -> App
registrations -> New registration*, name `stress-deploy`, single tenant, no
redirect URI). Note its **Application (client) ID** and **Directory (tenant)
ID**.

**b. Add the federated credential** (*the new app -> Certificates & secrets ->
Federated credentials -> Add credential*):

| Field | Value |
|---|---|
| Scenario | GitHub Actions deploying Azure resources |
| Organization | `onlyarya2004-maker` |
| Repository | `vision-stress-detection` |
| Entity type | **Environment** |
| Environment name | `azure-prod` |

That produces subject `repo:onlyarya2004-maker/vision-stress-detection:environment:azure-prod`,
issuer `https://token.actions.githubusercontent.com`, audience
`api://AzureADTokenExchange`. Because the subject names the *environment*, a
run that has not passed the approval gate cannot get a token.

**c. Give it the two narrow roles it needs** (not Contributor on the
subscription):

```powershell
$APPID = "<Application (client) ID from step a>"
$SP    = az ad sp list --filter "appId eq '$APPID'" --query "[0].id" -o tsv
az role assignment create --assignee $SP --role AcrPush `
    --scope $(az acr show -n $ACR -g $RG --query id -o tsv)
az role assignment create --assignee $SP --role Contributor `
    --scope $(az containerapp show -n $APP -g $RG --query id -o tsv)
```
`AcrPush` lets it push images but not delete the registry; `Contributor` scoped
to the single container app lets it change that app's image and nothing else -
it cannot create, delete or read anything elsewhere in the subscription.

**d. In GitHub** -> repo -> *Settings*:
- *Environments -> New environment* named `azure-prod`; tick **Required
  reviewers** and add yourself. (Optionally limit deployment branches to `dev`.)
- In that environment, add secrets `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`,
  `AZURE_SUBSCRIPTION_ID` and a variable `ACR_NAME`. Putting them on the
  *environment* rather than the repo means an unapproved job cannot read them.
- Copy the template to `.github/workflows/deploy.yml` and push.

Then: *Actions -> deploy-azure -> Run workflow -> ref `dev`* -> approve ->
it builds, pushes, rolls out and waits for `/_stcore/health` to answer.

Nothing triggers on `main`. The models are Git-LFS files, so each run downloads
~51 MB of LFS against GitHub's free 1 GB/month - roughly 20 deploys a month.
If app registration is blocked on the student tenant, skip all of this and run
the commands in 3b by hand (`docker build` -> `docker push` ->
`az containerapp update --image ...@<digest>`); CI is a convenience, not a requirement.

Supabase keep-alive: copy `deploy/github-workflow-keepalive.yml` to
`.github/workflows/` and add secrets `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`.

---------------------------------------------------------------------------
## 5. Demo-day checklist

- [ ] Day before: start the app and set `--min-replicas 1` (3b; the size is already
      4 vCPU / 8 GiB); Supabase project awake (open the dashboard once); run
      `scripts/rls_check.py` one last time. Stop the app again afterwards.
- [ ] **Two-user performance test - the one thing never measured locally.** A
      background browser tab gets throttled, so two real simultaneous clients
      could only ever be tested on two machines. On the hosted URL, with both
      laptops side by side:
      1. Both sign in. Confirm each sees only its own past sessions.
      2. Both press Start within a few seconds of each other and run 3 minutes.
      3. On each laptop watch the processed-frames-per-second readout. **Pass bar:
         >= 8 fps each, round trip <= ~1.5 s, no "falling behind" warnings.**
      4. During the run: the console logs in Log Analytics (3b - `az containerapp
         logs show` does not work on Express), and in the portal *Metrics* -> CPU
         usage / memory working set.
         Expect roughly 2.0-2.6 cores and ~1.3 GB total (measured: 632 MB per live
         session, 51 MB idle).
      5. Both press Stop. Both sessions must appear in their own Past sessions and
         in Supabase, with the row counts matching.
      If fps comes in below 8: set `POSE_EVERY_N_FRAMES = 3` in `code/config.py`
      and redeploy. That is pre-validated offline (behaviour-score r = 0.999,
      `models/pose_cadence_parity.txt`) and buys ~13-17 ms of the ~85 ms per frame.
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
  high**: the server is CPU-bound. It already has 4 vCPU (the Consumption maximum), so
  run Pose every 3rd frame:
  set `POSE_EVERY_N_FRAMES = 3` in `code/config.py` **only after**
  `python experiment_mediapipe_tasks_parity.py --mode pose-cadence --pose-every 3`
  shows the posture signals still agree (see the result recorded in
  `models/pose_cadence_parity.txt`).
- **Camera blocked**: the browser shows a message on the video. Click the camera icon in
  the address bar -> Allow -> "Try again". Camera access requires https (or localhost).
- **Page loads but stays on "Please wait" / WebSocket errors**: on the Express deployment
  the sign-in page loaded over the WebSocket with Streamlit's XSRF and CORS protection
  left ON (verified 2026-10-14), so do not switch them off. Check ingress first: port
  8501, transport HTTP/1.1, and the 404 note in 3a.
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

Since then: deployed to Azure Container Apps (3a); on 2026-10-14 the public address
served the sign-in page over HTTPS and the WebSocket, with no browser errors. Not yet
done against the hosted URL: a real sign-in, the two-laptop rehearsal, and a
real-webcam run.
