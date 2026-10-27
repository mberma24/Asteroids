# The Oracle VM was disabled: the Always Free A1 limit is now 2 OCPU / 12 GB

2026-10-27. `ml-trainer-2` (VM.Standard.A1.Flex, 4 OCPU / 24 GB, us-ashburn-1 AD-2, created
2026-08-20 during the free trial) was stopped by Oracle at 08:40 and 10:36 UTC and left
**disabled**. Every action request now fails with:

> Instance … is disabled and will not accept any action requests. Please contact customer
> support to reenable.

## Cause

The tenancy's Ampere A1 limit is **2 cores / 12 GB**, in every AD and regionally
(`standard-a1-core-count`, `standard-a1-core-regional-count`, `standard-a1-memory-count`,
`standard-a1-memory-regional-count`). The instance used 4 / 24, so resource availability read
`available: -2` cores and `-12` GB. The trial credits covered the excess; once the trial
ended the account was free-only and over the limit.

Oracle halved the Always Free A1 allowance on **2026-06-15**, from 3,000 OCPU-hours +
18,000 GB-hours/month (4 OCPU / 24 GB) to **1,500 OCPU-hours + 9,000 GB-hours/month
(2 OCPU / 12 GB)**, without announcement; free-tier users were told to resize by 2026-08-18.
The current documentation says: "All tenancies get the first 1,500 OCPU hours and 9,000 GB
hours per month for free … For Always Free tenancies, this is equivalent to 2 OCPUs and
12 GB of memory."

**The console banner is stale.** The Instances page still says 3,000 OCPU hours / 4 OCPUs /
24 GB. Trust the limits service (`oci limits value list --service-name compute`), not the
banner.

How the stop was traced: the audit log shows `ComputeApi.InstanceAction.end` with
`instanceActionType: stop` and no user principal, preceded by `ServiceApi` / `GetVnic` /
`DetachVnic` calls from Oracle-internal 172.24.x addresses -- Oracle did it, not a user.
Work requests for the instance were empty.

## What is still uncertain

- Pay-As-You-Go accounts: some Oracle support agents say they keep 4 / 24 free, others that
  overage is billed, and the documentation says 2 / 12 for all tenancies. Not tested here;
  the account stays free-only to avoid any charge.
- Whether a disabled instance can be terminated from the console (the CLI rejects resize and
  start).

## Consequences

- A stopped instance still counts against the core limit, so no new A1 VM can be created
  until the disabled one is terminated.
- An ephemeral public IP cannot be converted to reserved and is deleted on terminate, so the
  address changes once. The tenancy has 1 free reserved public IP; use it for the
  replacement so the address survives the next rebuild.
- Idle reclamation applies to free-only accounts: an instance is reclaimable if, over 7 days,
  95th-percentile CPU, network **and** (A1 only) memory are all under 20%. A JVM with
  `-Xms` = `-Xmx` and `AlwaysPreTouch` holding half the RAM keeps memory well above that.

## Recovery

1. Full boot-volume backup `ml-trainer-2-backup-20261027`
   (`ocid1.bootvolumebackup.oc1.iad.abuwcljroqhufsbrsm3axanimbb3lxg52xyd3dxbceld4ivap7ndwkhckyda`),
   47 GB, 21 GB unique. Taken while the instance was disabled; backups still work.
2. Terminate the disabled instance, **preserving** its boot volume.
3. Launch VM.Standard.A1.Flex at 2 OCPU / 12 GB from the preserved boot volume, same subnet,
   with a reserved public IP.

## Sources

- [Oracle: Always Free Resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)
- [Oracle: Public IP addresses](https://docs.oracle.com/en-us/iaas/Content/Network/Tasks/managingpublicIPs.htm)
- [InfoQ, 2026-07: Oracle Quietly Halves Free Tier Ampere A1 Compute Limits](https://www.infoq.com/news/2026/07/oracle-cloud-free-tier-limits/)
- [TerminalBytes: Oracle Cloud free tier 2026: 4 OCPU/24GB cut to 2 OCPU/12GB](https://terminalbytes.com/oracle-cloud-free-tier-changes-2026/)
- [Linuxiac: Oracle Quietly Cuts Free Tier Ampere A1 Resources in Half](https://linuxiac.com/oracle-quietly-cuts-free-tier-ampere-a1-resources-in-half/)
- [ETTAYEB: Oracle Halves Always Free ARM Limits, deadline 2026-08-18](https://ettayeb.fr/en/selfhosted/oracle-always-free-arm-limits-halved-2026/)
