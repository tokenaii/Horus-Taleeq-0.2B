# Hardware, Runtime, and Cost Record

Measurements were collected directly from the AWS training host.

| Component | Measured value |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU memory | 97,887 MiB (about 95.6 GiB) |
| Driver / CUDA | 595.91.07 / 13.2 |
| CPU | Intel Xeon Platinum 8559C |
| Physical / logical CPUs | 4 / 8 |
| System RAM | 62 GiB reported |
| Swap | none |
| Data volume | 1.8 TB NVMe mount |
| OS | Amazon Linux 2023, x86_64 |

The EC2 instance family was not available through instance metadata during the
audit, so an hourly price is intentionally not guessed. A defensible rate requires
the exact instance type, region, purchase model, and date.

```text
training_cost = measured_wall_clock_hours × region_price_per_hour
```

The two core pretraining runs consumed 28.6724 wall-clock hours combined. This is
a runtime measurement, not a billing statement.
