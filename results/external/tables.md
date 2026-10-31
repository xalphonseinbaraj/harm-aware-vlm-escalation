# Revision analysis (theta=0.5, tau=0.5)

## Table A. Escalation policies (mean over n test images; IoU / Dice / Boundary-F1 / HD95 px; VLM calls; latency per image)

| Run | Policy | IoU | Dice | BF1 | HD95 | VLM calls | ms/image |
|---|---|---|---|---|---|---|---|
| CVC-ClinicDB_segformer_s0__medgemma4b_p1 | never | 0.769 | 0.838 | 0.530 | 33.7 | 0/62 | 26 |
| CVC-ClinicDB_segformer_s0__medgemma4b_p1 | always | 0.725 | 0.813 | 0.223 | 34.7 | 62/62 | 4538 |
| CVC-ClinicDB_segformer_s0__medgemma4b_p1 | always+V | 0.750 | 0.828 | 0.369 | 34.0 | 62/62 | 4627 |
| CVC-ClinicDB_segformer_s0__medgemma4b_p1 | conf-only | 0.761 | 0.834 | 0.457 | 33.6 | 28/62 | 2065 |
| CVC-ClinicDB_segformer_s0__medgemma4b_p1 | gate | 0.759 | 0.833 | 0.461 | 33.9 | 28/62 | 2065 |
| CVC-ClinicDB_segformer_s0__medgemma4b_p1 | gate+V | 0.766 | 0.837 | 0.493 | 33.8 | 28/62 | 2105 |
| CVC-ClinicDB_segformer_s0__medgemma4b_p1 | oracle-diff | 0.767 | 0.838 | 0.503 | 33.6 | 14/62 | 1047 |
| CVC-ClinicDB_segformer_s0__medgemma4b_p1 | oracle-gain | 0.771 | 0.841 | 0.523 | 33.6 | 9/62 | 683 |
| CVC-ClinicDB_segformer_s0__medgemma4b_p1 | random (same rate, 10k draws) | 0.749 [0.745, 0.753] | | | | 28/62 | |
| CVC-ClinicDB_strong_s0__qwen7b_p3 | never | 0.533 | 0.626 | 0.277 | 89.6 | 0/62 | 25 |
| CVC-ClinicDB_strong_s0__qwen7b_p3 | always | 0.540 | 0.631 | 0.298 | 89.0 | 62/62 | 10062 |
| CVC-ClinicDB_strong_s0__qwen7b_p3 | always+V | 0.539 | 0.630 | 0.297 | 89.1 | 62/62 | 10079 |
| CVC-ClinicDB_strong_s0__qwen7b_p3 | conf-only | 0.537 | 0.629 | 0.293 | 89.4 | 32/62 | 5207 |
| CVC-ClinicDB_strong_s0__qwen7b_p3 | gate | 0.538 | 0.630 | 0.291 | 89.0 | 32/62 | 5207 |
| CVC-ClinicDB_strong_s0__qwen7b_p3 | gate+V | 0.538 | 0.630 | 0.291 | 89.0 | 32/62 | 5215 |
| CVC-ClinicDB_strong_s0__qwen7b_p3 | oracle-diff | 0.538 | 0.630 | 0.292 | 89.1 | 37/62 | 6016 |
| CVC-ClinicDB_strong_s0__qwen7b_p3 | oracle-gain | 0.542 | 0.634 | 0.298 | 89.1 | 7/62 | 1161 |
| CVC-ClinicDB_strong_s0__qwen7b_p3 | random (same rate, 10k draws) | 0.536 [0.533, 0.540] | | | | 32/62 | |
| CVC-ClinicDB_strong_s1__qwen7b_p3 | never | 0.582 | 0.668 | 0.343 | 96.3 | 0/62 | 24 |
| CVC-ClinicDB_strong_s1__qwen7b_p3 | always | 0.586 | 0.669 | 0.357 | 94.2 | 62/62 | 10067 |
| CVC-ClinicDB_strong_s1__qwen7b_p3 | always+V | 0.586 | 0.669 | 0.357 | 94.2 | 62/62 | 10088 |
| CVC-ClinicDB_strong_s1__qwen7b_p3 | conf-only | 0.585 | 0.670 | 0.347 | 94.2 | 32/62 | 5209 |
| CVC-ClinicDB_strong_s1__qwen7b_p3 | gate | 0.585 | 0.668 | 0.349 | 94.2 | 32/62 | 5209 |
| CVC-ClinicDB_strong_s1__qwen7b_p3 | gate+V | 0.585 | 0.668 | 0.349 | 94.2 | 32/62 | 5220 |
| CVC-ClinicDB_strong_s1__qwen7b_p3 | oracle-diff | 0.583 | 0.667 | 0.341 | 96.5 | 31/62 | 5047 |
| CVC-ClinicDB_strong_s1__qwen7b_p3 | oracle-gain | 0.590 | 0.673 | 0.358 | 93.8 | 9/62 | 1485 |
| CVC-ClinicDB_strong_s1__qwen7b_p3 | random (same rate, 10k draws) | 0.584 [0.580, 0.588] | | | | 32/62 | |
| CVC-ClinicDB_strong_s2__qwen7b_p3 | never | 0.512 | 0.610 | 0.286 | 99.3 | 0/62 | 25 |
| CVC-ClinicDB_strong_s2__qwen7b_p3 | always | 0.517 | 0.612 | 0.295 | 99.0 | 62/62 | 10231 |
| CVC-ClinicDB_strong_s2__qwen7b_p3 | always+V | 0.517 | 0.612 | 0.295 | 99.0 | 62/62 | 10256 |
| CVC-ClinicDB_strong_s2__qwen7b_p3 | conf-only | 0.518 | 0.613 | 0.296 | 99.0 | 36/62 | 5953 |
| CVC-ClinicDB_strong_s2__qwen7b_p3 | gate | 0.515 | 0.611 | 0.291 | 99.0 | 36/62 | 5953 |
| CVC-ClinicDB_strong_s2__qwen7b_p3 | gate+V | 0.515 | 0.611 | 0.291 | 99.0 | 36/62 | 5967 |
| CVC-ClinicDB_strong_s2__qwen7b_p3 | oracle-diff | 0.513 | 0.609 | 0.285 | 99.2 | 38/62 | 6282 |
| CVC-ClinicDB_strong_s2__qwen7b_p3 | oracle-gain | 0.520 | 0.615 | 0.300 | 98.8 | 9/62 | 1510 |
| CVC-ClinicDB_strong_s2__qwen7b_p3 | random (same rate, 10k draws) | 0.515 [0.512, 0.518] | | | | 36/62 | |
| CVC-ColonDB_segformer_s0__medgemma4b_p1 | never | 0.698 | 0.777 | 0.406 | 49.6 | 0/380 | 26 |
| CVC-ColonDB_segformer_s0__medgemma4b_p1 | always | 0.669 | 0.759 | 0.237 | 51.2 | 380/380 | 4494 |
| CVC-ColonDB_segformer_s0__medgemma4b_p1 | always+V | 0.688 | 0.771 | 0.339 | 50.7 | 380/380 | 4582 |
| CVC-ColonDB_segformer_s0__medgemma4b_p1 | conf-only | 0.689 | 0.771 | 0.344 | 50.2 | 217/380 | 2578 |
| CVC-ColonDB_segformer_s0__medgemma4b_p1 | gate | 0.687 | 0.770 | 0.333 | 50.3 | 217/380 | 2578 |
| CVC-ColonDB_segformer_s0__medgemma4b_p1 | gate+V | 0.695 | 0.775 | 0.376 | 50.1 | 217/380 | 2629 |
| CVC-ColonDB_segformer_s0__medgemma4b_p1 | oracle-diff | 0.697 | 0.776 | 0.386 | 49.9 | 119/380 | 1427 |
| CVC-ColonDB_segformer_s0__medgemma4b_p1 | oracle-gain | 0.702 | 0.780 | 0.404 | 49.4 | 73/380 | 886 |
| CVC-ColonDB_segformer_s0__medgemma4b_p1 | random (same rate, 10k draws) | 0.681 [0.680, 0.683] | | | | 217/380 | |
| CVC-ColonDB_strong_s0__medgemma4b_p1 | never | 0.449 | 0.533 | 0.207 | 149.3 | 0/380 | 10 |
| CVC-ColonDB_strong_s0__medgemma4b_p1 | always | 0.433 | 0.525 | 0.131 | 149.5 | 380/380 | 4561 |
| CVC-ColonDB_strong_s0__medgemma4b_p1 | always+V | 0.433 | 0.525 | 0.134 | 149.5 | 380/380 | 4592 |
| CVC-ColonDB_strong_s0__medgemma4b_p1 | conf-only | 0.447 | 0.534 | 0.186 | 149.0 | 181/380 | 2179 |
| CVC-ColonDB_strong_s0__medgemma4b_p1 | gate | 0.448 | 0.534 | 0.188 | 149.0 | 181/380 | 2179 |
| CVC-ColonDB_strong_s0__medgemma4b_p1 | gate+V | 0.448 | 0.534 | 0.190 | 149.0 | 181/380 | 2194 |
| CVC-ColonDB_strong_s0__medgemma4b_p1 | oracle-diff | 0.450 | 0.536 | 0.191 | 149.4 | 243/380 | 2921 |
| CVC-ColonDB_strong_s0__medgemma4b_p1 | oracle-gain | 0.457 | 0.542 | 0.200 | 148.6 | 116/380 | 1401 |
| CVC-ColonDB_strong_s0__medgemma4b_p1 | random (same rate, 10k draws) | 0.441 [0.439, 0.443] | | | | 181/380 | |
| CVC-ColonDB_strong_s0__qwen7b_p3 | never | 0.449 | 0.533 | 0.207 | 149.3 | 0/380 | 26 |
| CVC-ColonDB_strong_s0__qwen7b_p3 | always | 0.449 | 0.533 | 0.212 | 148.8 | 380/380 | 10159 |
| CVC-ColonDB_strong_s0__qwen7b_p3 | always+V | 0.449 | 0.533 | 0.211 | 148.8 | 380/380 | 10167 |
| CVC-ColonDB_strong_s0__qwen7b_p3 | conf-only | 0.449 | 0.534 | 0.209 | 149.2 | 181/380 | 4853 |
| CVC-ColonDB_strong_s0__qwen7b_p3 | gate | 0.449 | 0.533 | 0.210 | 149.2 | 181/380 | 4853 |
| CVC-ColonDB_strong_s0__qwen7b_p3 | gate+V | 0.449 | 0.533 | 0.209 | 149.1 | 181/380 | 4857 |
| CVC-ColonDB_strong_s0__qwen7b_p3 | oracle-diff | 0.449 | 0.534 | 0.211 | 149.2 | 243/380 | 6506 |
| CVC-ColonDB_strong_s0__qwen7b_p3 | oracle-gain | 0.451 | 0.535 | 0.214 | 148.6 | 16/380 | 454 |
| CVC-ColonDB_strong_s0__qwen7b_p3 | random (same rate, 10k draws) | 0.449 [0.448, 0.449] | | | | 181/380 | |
| CVC-ColonDB_strong_s1__qwen7b_p3 | never | 0.486 | 0.570 | 0.255 | 141.4 | 0/380 | 26 |
| CVC-ColonDB_strong_s1__qwen7b_p3 | always | 0.486 | 0.569 | 0.257 | 141.5 | 380/380 | 10038 |
| CVC-ColonDB_strong_s1__qwen7b_p3 | always+V | 0.486 | 0.569 | 0.257 | 141.5 | 380/380 | 10048 |
| CVC-ColonDB_strong_s1__qwen7b_p3 | conf-only | 0.486 | 0.570 | 0.255 | 141.5 | 192/380 | 5085 |
| CVC-ColonDB_strong_s1__qwen7b_p3 | gate | 0.485 | 0.569 | 0.256 | 141.5 | 192/380 | 5085 |
| CVC-ColonDB_strong_s1__qwen7b_p3 | gate+V | 0.485 | 0.569 | 0.256 | 141.5 | 192/380 | 5091 |
| CVC-ColonDB_strong_s1__qwen7b_p3 | oracle-diff | 0.486 | 0.569 | 0.254 | 141.5 | 226/380 | 5981 |
| CVC-ColonDB_strong_s1__qwen7b_p3 | oracle-gain | 0.488 | 0.572 | 0.261 | 141.3 | 20/380 | 555 |
| CVC-ColonDB_strong_s1__qwen7b_p3 | random (same rate, 10k draws) | 0.486 [0.485, 0.487] | | | | 192/380 | |
| CVC-ColonDB_strong_s2__qwen7b_p3 | never | 0.422 | 0.502 | 0.207 | 177.3 | 0/380 | 26 |
| CVC-ColonDB_strong_s2__qwen7b_p3 | always | 0.423 | 0.502 | 0.211 | 177.1 | 380/380 | 10005 |
| CVC-ColonDB_strong_s2__qwen7b_p3 | always+V | 0.423 | 0.502 | 0.211 | 177.1 | 380/380 | 10017 |
| CVC-ColonDB_strong_s2__qwen7b_p3 | conf-only | 0.423 | 0.502 | 0.210 | 177.3 | 146/380 | 3861 |
| CVC-ColonDB_strong_s2__qwen7b_p3 | gate | 0.422 | 0.502 | 0.209 | 177.3 | 146/380 | 3861 |
| CVC-ColonDB_strong_s2__qwen7b_p3 | gate+V | 0.422 | 0.502 | 0.209 | 177.3 | 146/380 | 3866 |
| CVC-ColonDB_strong_s2__qwen7b_p3 | oracle-diff | 0.422 | 0.502 | 0.209 | 177.1 | 257/380 | 6776 |
| CVC-ColonDB_strong_s2__qwen7b_p3 | oracle-gain | 0.425 | 0.504 | 0.214 | 177.0 | 18/380 | 500 |
| CVC-ColonDB_strong_s2__qwen7b_p3 | random (same rate, 10k draws) | 0.422 [0.421, 0.423] | | | | 146/380 | |
| ETIS-LaribPolypDB_segformer_s0__medgemma4b_p1 | never | 0.670 | 0.748 | 0.482 | 59.0 | 0/196 | 25 |
| ETIS-LaribPolypDB_segformer_s0__medgemma4b_p1 | always | 0.627 | 0.721 | 0.255 | 60.3 | 196/196 | 4553 |
| ETIS-LaribPolypDB_segformer_s0__medgemma4b_p1 | always+V | 0.641 | 0.729 | 0.321 | 60.0 | 196/196 | 4640 |
| ETIS-LaribPolypDB_segformer_s0__medgemma4b_p1 | conf-only | 0.645 | 0.730 | 0.376 | 59.6 | 149/196 | 3468 |
| ETIS-LaribPolypDB_segformer_s0__medgemma4b_p1 | gate | 0.643 | 0.730 | 0.370 | 59.6 | 149/196 | 3468 |
| ETIS-LaribPolypDB_segformer_s0__medgemma4b_p1 | gate+V | 0.654 | 0.736 | 0.408 | 59.4 | 149/196 | 3534 |
| ETIS-LaribPolypDB_segformer_s0__medgemma4b_p1 | oracle-diff | 0.665 | 0.743 | 0.457 | 59.4 | 71/196 | 1666 |
| ETIS-LaribPolypDB_segformer_s0__medgemma4b_p1 | oracle-gain | 0.673 | 0.750 | 0.487 | 59.0 | 22/196 | 535 |
| ETIS-LaribPolypDB_segformer_s0__medgemma4b_p1 | random (same rate, 10k draws) | 0.638 [0.635, 0.640] | | | | 149/196 | |
| ETIS-LaribPolypDB_strong_s0__qwen7b_p3 | never | 0.335 | 0.407 | 0.217 | 205.4 | 0/196 | 25 |
| ETIS-LaribPolypDB_strong_s0__qwen7b_p3 | always | 0.337 | 0.408 | 0.221 | 205.2 | 196/196 | 9973 |
| ETIS-LaribPolypDB_strong_s0__qwen7b_p3 | always+V | 0.337 | 0.408 | 0.221 | 205.2 | 196/196 | 9985 |
| ETIS-LaribPolypDB_strong_s0__qwen7b_p3 | conf-only | 0.336 | 0.408 | 0.220 | 205.2 | 112/196 | 5710 |
| ETIS-LaribPolypDB_strong_s0__qwen7b_p3 | gate | 0.338 | 0.409 | 0.222 | 205.2 | 112/196 | 5710 |
| ETIS-LaribPolypDB_strong_s0__qwen7b_p3 | gate+V | 0.337 | 0.408 | 0.222 | 205.2 | 112/196 | 5718 |
| ETIS-LaribPolypDB_strong_s0__qwen7b_p3 | oracle-diff | 0.337 | 0.408 | 0.220 | 205.2 | 156/196 | 7943 |
| ETIS-LaribPolypDB_strong_s0__qwen7b_p3 | oracle-gain | 0.339 | 0.410 | 0.225 | 205.2 | 18/196 | 941 |
| ETIS-LaribPolypDB_strong_s0__qwen7b_p3 | random (same rate, 10k draws) | 0.336 [0.335, 0.337] | | | | 112/196 | |
| ETIS-LaribPolypDB_strong_s1__qwen7b_p3 | never | 0.378 | 0.453 | 0.257 | 160.0 | 0/196 | 25 |
| ETIS-LaribPolypDB_strong_s1__qwen7b_p3 | always | 0.379 | 0.454 | 0.259 | 159.6 | 196/196 | 10013 |
| ETIS-LaribPolypDB_strong_s1__qwen7b_p3 | always+V | 0.379 | 0.454 | 0.259 | 159.6 | 196/196 | 10025 |
| ETIS-LaribPolypDB_strong_s1__qwen7b_p3 | conf-only | 0.379 | 0.454 | 0.258 | 159.7 | 137/196 | 7008 |
| ETIS-LaribPolypDB_strong_s1__qwen7b_p3 | gate | 0.379 | 0.454 | 0.259 | 159.6 | 137/196 | 7008 |
| ETIS-LaribPolypDB_strong_s1__qwen7b_p3 | gate+V | 0.379 | 0.454 | 0.259 | 159.6 | 137/196 | 7016 |
| ETIS-LaribPolypDB_strong_s1__qwen7b_p3 | oracle-diff | 0.378 | 0.453 | 0.257 | 159.8 | 142/196 | 7262 |
| ETIS-LaribPolypDB_strong_s1__qwen7b_p3 | oracle-gain | 0.380 | 0.455 | 0.261 | 159.7 | 16/196 | 843 |
| ETIS-LaribPolypDB_strong_s1__qwen7b_p3 | random (same rate, 10k draws) | 0.379 [0.378, 0.380] | | | | 137/196 | |
| ETIS-LaribPolypDB_strong_s2__qwen7b_p3 | never | 0.314 | 0.390 | 0.186 | 205.2 | 0/196 | 24 |
| ETIS-LaribPolypDB_strong_s2__qwen7b_p3 | always | 0.316 | 0.390 | 0.186 | 204.5 | 196/196 | 10011 |
| ETIS-LaribPolypDB_strong_s2__qwen7b_p3 | always+V | 0.316 | 0.390 | 0.186 | 204.5 | 196/196 | 10024 |
| ETIS-LaribPolypDB_strong_s2__qwen7b_p3 | conf-only | 0.315 | 0.390 | 0.184 | 204.6 | 136/196 | 6955 |
| ETIS-LaribPolypDB_strong_s2__qwen7b_p3 | gate | 0.316 | 0.390 | 0.186 | 204.5 | 136/196 | 6955 |
| ETIS-LaribPolypDB_strong_s2__qwen7b_p3 | gate+V | 0.316 | 0.390 | 0.186 | 204.5 | 136/196 | 6964 |
| ETIS-LaribPolypDB_strong_s2__qwen7b_p3 | oracle-diff | 0.316 | 0.390 | 0.185 | 204.5 | 165/196 | 8432 |
| ETIS-LaribPolypDB_strong_s2__qwen7b_p3 | oracle-gain | 0.317 | 0.392 | 0.190 | 204.8 | 11/196 | 587 |
| ETIS-LaribPolypDB_strong_s2__qwen7b_p3 | random (same rate, 10k draws) | 0.315 [0.315, 0.316] | | | | 136/196 | |

## Table B. Paired Wilcoxon signed-rank tests on per-image IoU

| Run | gate+V vs always | gate+V vs always+V | gate vs always | gate+V vs never | gate+V vs gate | always vs never |
|---|---|---|---|---|---|---|
| CVC-ClinicDB_segformer_s0__medgemma4b_p1 | <0.0001 | 0.0004 | <0.0001 | 0.2553 | 0.0058 | <0.0001 |
| CVC-ClinicDB_strong_s0__qwen7b_p3 | 0.2850 | 0.6547 | 0.2850 | 0.2135 | 1.0000 | 0.0995 |
| CVC-ClinicDB_strong_s1__qwen7b_p3 | 0.1797 | 0.1797 | 0.1797 | 0.5936 | 1.0000 | 0.3259 |
| CVC-ClinicDB_strong_s2__qwen7b_p3 | 0.1797 | 0.1797 | 0.1797 | 0.3343 | 1.0000 | 0.1359 |
| CVC-ColonDB_segformer_s0__medgemma4b_p1 | <0.0001 | <0.0001 | <0.0001 | 0.0016 | <0.0001 | <0.0001 |
| CVC-ColonDB_strong_s0__medgemma4b_p1 | <0.0001 | <0.0001 | <0.0001 | 0.6081 | 0.3139 | <0.0001 |
| CVC-ColonDB_strong_s0__qwen7b_p3 | 0.6378 | 0.5937 | 0.5937 | 0.8721 | 1.0000 | 0.9929 |
| CVC-ColonDB_strong_s1__qwen7b_p3 | 0.2131 | 0.2131 | 0.2131 | 0.6041 | 1.0000 | 0.8700 |
| CVC-ColonDB_strong_s2__qwen7b_p3 | 0.8767 | 0.8767 | 0.8767 | 0.4802 | 1.0000 | 0.5210 |
| ETIS-LaribPolypDB_segformer_s0__medgemma4b_p1 | <0.0001 | <0.0001 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| ETIS-LaribPolypDB_strong_s0__qwen7b_p3 | 1.0000 | 0.5930 | 0.5930 | 0.0262 | 0.3173 | 0.0361 |
| ETIS-LaribPolypDB_strong_s1__qwen7b_p3 | 1.0000 | 1.0000 | 1.0000 | 0.0416 | 1.0000 | 0.0416 |
| ETIS-LaribPolypDB_strong_s2__qwen7b_p3 | 1.0000 | 1.0000 | 1.0000 | 0.1451 | 1.0000 | 0.1451 |

## Table C. Gate vs. true difficulty (backbone IoU < 0.7)  [Reviewer 3, Results 1]

| Run | esc. rate | hard rate | Gate P | Gate R | Gate F1 | Conf-only P / R / F1 | Random P (95% CI) | AUROC gate | AUROC conf | P(random ≥ gate IoU) |
|---|---|---|---|---|---|---|---|---|---|---|
| CVC-ClinicDB_segformer_s0__medgemma4b_p1 | 0.452 | 0.226 | 0.429 | 0.857 | 0.571 | 0.500 / 1.000 / 0.667 | 0.226 (0.107–0.357) | 0.876 | 0.932 | 0.000 |
| CVC-ClinicDB_strong_s0__qwen7b_p3 | 0.516 | 0.597 | 0.938 | 0.811 | 0.870 | 0.750 / 0.649 / 0.696 | 0.597 (0.469–0.719) | 0.909 | 0.690 | 0.198 |
| CVC-ClinicDB_strong_s1__qwen7b_p3 | 0.516 | 0.500 | 0.750 | 0.774 | 0.762 | 0.562 / 0.581 / 0.571 | 0.500 (0.375–0.625) | 0.829 | 0.661 | 0.367 |
| CVC-ClinicDB_strong_s2__qwen7b_p3 | 0.581 | 0.613 | 0.861 | 0.816 | 0.838 | 0.722 / 0.684 / 0.703 | 0.613 (0.500–0.722) | 0.865 | 0.693 | 0.484 |
| CVC-ColonDB_segformer_s0__medgemma4b_p1 | 0.571 | 0.313 | 0.484 | 0.882 | 0.625 | 0.525 / 0.958 / 0.679 | 0.313 (0.272–0.355) | 0.876 | 0.924 | 0.000 |
| CVC-ColonDB_strong_s0__medgemma4b_p1 | 0.476 | 0.639 | 0.818 | 0.609 | 0.698 | 0.834 / 0.621 / 0.712 | 0.639 (0.586–0.691) | 0.654 | 0.787 | 0.000 |
| CVC-ColonDB_strong_s0__qwen7b_p3 | 0.476 | 0.639 | 0.818 | 0.609 | 0.698 | 0.834 / 0.621 / 0.712 | 0.640 (0.591–0.691) | 0.654 | 0.787 | 0.597 |
| CVC-ColonDB_strong_s1__qwen7b_p3 | 0.505 | 0.595 | 0.729 | 0.619 | 0.670 | 0.646 / 0.549 / 0.593 | 0.595 (0.547–0.646) | 0.651 | 0.581 | 0.685 |
| CVC-ColonDB_strong_s2__qwen7b_p3 | 0.384 | 0.676 | 0.890 | 0.506 | 0.645 | 0.884 / 0.502 / 0.640 | 0.676 (0.616–0.740) | 0.636 | 0.748 | 0.144 |
| ETIS-LaribPolypDB_segformer_s0__medgemma4b_p1 | 0.760 | 0.362 | 0.470 | 0.986 | 0.636 | 0.477 / 1.000 / 0.645 | 0.362 (0.322–0.396) | 0.881 | 0.936 | 0.000 |
| ETIS-LaribPolypDB_strong_s0__qwen7b_p3 | 0.571 | 0.796 | 0.830 | 0.596 | 0.694 | 0.795 / 0.571 / 0.664 | 0.797 (0.750–0.848) | 0.592 | 0.583 | 0.016 |
| ETIS-LaribPolypDB_strong_s1__qwen7b_p3 | 0.699 | 0.724 | 0.759 | 0.732 | 0.746 | 0.781 / 0.754 / 0.767 | 0.725 (0.686–0.766) | 0.679 | 0.632 | 0.100 |
| ETIS-LaribPolypDB_strong_s2__qwen7b_p3 | 0.694 | 0.842 | 0.853 | 0.703 | 0.771 | 0.853 / 0.703 / 0.771 | 0.842 (0.809–0.875) | 0.667 | 0.665 | 0.117 |

## Table D. Confidence-head calibration  [Reviewer 3, Related work 1]

| Run | ECE | MAE | Pearson r | Spearman ρ | conf range |
|---|---|---|---|---|---|
| CVC-ClinicDB_segformer_s0__medgemma4b_p1 | 0.198 | 0.198 | 0.801 | 0.791 | 0.819–0.989 |
| CVC-ClinicDB_strong_s0__qwen7b_p3 | 0.178 | 0.273 | 0.286 | 0.331 | 0.558–0.830 |
| CVC-ClinicDB_strong_s1__qwen7b_p3 | 0.167 | 0.264 | 0.250 | 0.275 | 0.624–0.860 |
| CVC-ClinicDB_strong_s2__qwen7b_p3 | 0.122 | 0.270 | 0.332 | 0.328 | 0.517–0.817 |
| CVC-ColonDB_segformer_s0__medgemma4b_p1 | 0.262 | 0.262 | 0.756 | 0.799 | 0.729–0.988 |
| CVC-ColonDB_strong_s0__medgemma4b_p1 | 0.282 | 0.322 | 0.389 | 0.414 | 0.550–0.841 |
| CVC-ColonDB_strong_s0__qwen7b_p3 | 0.282 | 0.322 | 0.389 | 0.414 | 0.550–0.841 |
| CVC-ColonDB_strong_s1__qwen7b_p3 | 0.203 | 0.310 | 0.185 | 0.176 | 0.476–0.827 |
| CVC-ColonDB_strong_s2__qwen7b_p3 | 0.202 | 0.313 | 0.448 | 0.446 | 0.428–0.773 |
| ETIS-LaribPolypDB_segformer_s0__medgemma4b_p1 | 0.274 | 0.274 | 0.711 | 0.875 | 0.598–0.991 |
| ETIS-LaribPolypDB_strong_s0__qwen7b_p3 | 0.370 | 0.420 | 0.109 | -0.001 | 0.510–0.893 |
| ETIS-LaribPolypDB_strong_s1__qwen7b_p3 | 0.275 | 0.347 | 0.226 | 0.202 | 0.412–0.901 |
| ETIS-LaribPolypDB_strong_s2__qwen7b_p3 | 0.247 | 0.333 | 0.258 | 0.139 | 0.369–0.836 |

## Table E. Why indiscriminate escalation hurts: effect of the VLM edit by difficulty, and VLM behaviour

| Run | ΔIoU easy | % easy hurt | ΔIoU hard | % hard helped | % hard hurt | adjustments (all images) | template-copy rate | verif. accept rate | F vs helped (r, p) |
|---|---|---|---|---|---|---|---|---|---|
| CVC-ClinicDB_segformer_s0__medgemma4b_p1 | -0.054 | 94% | -0.006 | 50% | 50% | {'expand': 61, 'shift_boundary': 1} | 0% | 0.541 | 0.232, 0.0720 |
| CVC-ClinicDB_strong_s0__qwen7b_p3 | 0.005 | 0% | 0.009 | 14% | 14% | {'no_change': 50, 'shrink': 12} | 76% | 0.917 | -0.314, 0.3206 |
| CVC-ClinicDB_strong_s1__qwen7b_p3 | 0.007 | 0% | 0.002 | 13% | 23% | {'shrink': 16, 'no_change': 46} | 69% | 1.000 | -0.228, 0.3951 |
| CVC-ClinicDB_strong_s2__qwen7b_p3 | 0.010 | 0% | 0.002 | 13% | 21% | {'shrink': 18, 'no_change': 44} | 71% | 1.000 | 0.075, 0.7660 |
| CVC-ColonDB_segformer_s0__medgemma4b_p1 | -0.042 | 89% | -0.003 | 38% | 54% | {'expand': 369, 'shift_boundary': 5, 'shrink': 6} | 0% | 0.472 | 0.268, <0.0001 |
| CVC-ColonDB_strong_s0__medgemma4b_p1 | -0.047 | 88% | 0.002 | 41% | 25% | {'shrink': 22, 'expand': 346, 'shift_boundary': 12} | 0% | 0.956 | 0.202, 0.0003 |
| CVC-ColonDB_strong_s0__qwen7b_p3 | -0.001 | 4% | 0.000 | 5% | 5% | {'no_change': 343, 'shrink': 37} | 77% | 0.917 | -0.200, 0.2417 |
| CVC-ColonDB_strong_s1__qwen7b_p3 | 0.001 | 3% | -0.000 | 6% | 9% | {'no_change': 333, 'shrink': 46, 'expand': 1} | 78% | 1.000 | -0.125, 0.4081 |
| CVC-ColonDB_strong_s2__qwen7b_p3 | 0.001 | 3% | 0.001 | 5% | 9% | {'no_change': 327, 'shrink': 53} | 74% | 1.000 | 0.144, 0.3097 |
| ETIS-LaribPolypDB_segformer_s0__medgemma4b_p1 | -0.060 | 98% | -0.014 | 27% | 52% | {'expand': 187, 'shift_boundary': 6, 'shrink': 3} | 0% | 0.709 | 0.147, 0.0396 |
| ETIS-LaribPolypDB_strong_s0__qwen7b_p3 | 0.003 | 2% | 0.002 | 9% | 4% | {'no_change': 166, 'shrink': 30} | 72% | 0.967 | -0.129, 0.4976 |
| ETIS-LaribPolypDB_strong_s1__qwen7b_p3 | 0.006 | 0% | 0.000 | 5% | 5% | {'no_change': 170, 'shrink': 26} | 79% | 1.000 | 0.090, 0.6611 |
| ETIS-LaribPolypDB_strong_s2__qwen7b_p3 | 0.001 | 3% | 0.002 | 6% | 7% | {'shrink': 30, 'no_change': 166} | 76% | 1.000 | 0.257, 0.1698 |

## Table F. Timing breakdown (ms, mean per image)

| Run | backbone | gate | overlay | VLM call | verification |
|---|---|---|---|---|---|
| CVC-ClinicDB_segformer_s0__medgemma4b_p1 | 25.6 | 2.79 | 52.5 | 4457 | 89.2 |
| CVC-ClinicDB_strong_s0__qwen7b_p3 | 24.6 | 3.34 | 54.1 | 9980 | 16.3 |
| CVC-ClinicDB_strong_s1__qwen7b_p3 | 24.3 | 3.21 | 59.8 | 9979 | 21.5 |
| CVC-ClinicDB_strong_s2__qwen7b_p3 | 25.3 | 3.36 | 57.2 | 10145 | 24.8 |
| CVC-ColonDB_segformer_s0__medgemma4b_p1 | 25.7 | 2.69 | 59.7 | 4406 | 88.2 |
| CVC-ColonDB_strong_s0__medgemma4b_p1 | 10.2 | 2.21 | 40.7 | 4507 | 31.0 |
| CVC-ColonDB_strong_s0__qwen7b_p3 | 25.6 | 2.22 | 37.3 | 10094 | 8.2 |
| CVC-ColonDB_strong_s1__qwen7b_p3 | 25.5 | 2.29 | 41.5 | 9968 | 10.3 |
| CVC-ColonDB_strong_s2__qwen7b_p3 | 25.5 | 2.34 | 37.8 | 9940 | 11.6 |
| ETIS-LaribPolypDB_segformer_s0__medgemma4b_p1 | 24.7 | 2.10 | 44.6 | 4482 | 87.3 |
| ETIS-LaribPolypDB_strong_s0__qwen7b_p3 | 25.0 | 2.51 | 29.3 | 9916 | 12.8 |
| ETIS-LaribPolypDB_strong_s1__qwen7b_p3 | 24.5 | 3.13 | 44.6 | 9941 | 11.4 |
| ETIS-LaribPolypDB_strong_s2__qwen7b_p3 | 24.5 | 2.47 | 41.6 | 9942 | 12.9 |

## Table G. Pooled across seeds (mean ± SD of per-seed means; pooled Wilcoxon over all image-seed pairs)

| Backbone / VLM | seeds | never | always | gate+V (proposed) | esc. rate | gate+V vs always p | gate+V vs never p |
|---|---|---|---|---|---|---|---|
| CVC-ClinicDB_segformer / medgemma4b_p1 | 1 | 0.769 ± 0.000 | 0.725 ± 0.000 | 0.766 ± 0.000 | 0.452 | <0.0001 | 0.2553 |
| CVC-ClinicDB_strong / qwen7b_p3 | 3 | 0.542 ± 0.036 | 0.548 ± 0.035 | 0.546 ± 0.035 | 0.538 | 0.0280 | 0.1059 |
| CVC-ColonDB_segformer / medgemma4b_p1 | 1 | 0.698 ± 0.000 | 0.669 ± 0.000 | 0.695 ± 0.000 | 0.571 | <0.0001 | 0.0016 |
| CVC-ColonDB_strong / medgemma4b_p1 | 1 | 0.449 ± 0.000 | 0.433 ± 0.000 | 0.448 ± 0.000 | 0.476 | <0.0001 | 0.6081 |
| CVC-ColonDB_strong / qwen7b_p3 | 3 | 0.452 ± 0.032 | 0.452 ± 0.032 | 0.452 ± 0.032 | 0.455 | 0.3286 | 0.9746 |
| ETIS-LaribPolypDB_segformer / medgemma4b_p1 | 1 | 0.670 ± 0.000 | 0.627 ± 0.000 | 0.654 ± 0.000 | 0.760 | <0.0001 | <0.0001 |
| ETIS-LaribPolypDB_strong / qwen7b_p3 | 3 | 0.342 ± 0.032 | 0.344 ± 0.032 | 0.344 ± 0.032 | 0.655 | 1.0000 | 0.0006 |

## Table H. Matched baselines (same split / resolution / augmentation / model selection)  [Reviewer 1, c6]

| Model | IoU | Dice | BF1 | HD95 | Params (M) | GMACs | ms/img |
|---|---|---|---|---|---|---|---|
| deeplabv3p-resnet50 (base_CVC-ClinicDB_deeplabv3p_r50) | 0.721 | 0.790 | 0.494 | 63.2 | – | – | – |
| fpn-tu-pvt_v2_b2 (base_CVC-ClinicDB_fpn_pvtv2b2) | 0.772 | 0.839 | 0.544 | 34.8 | – | – | – |
| segformer-mit_b2 (base_CVC-ClinicDB_segformer_mitb2) | 0.769 | 0.838 | 0.530 | 33.7 | – | – | – |
| unet-resnet34 (base_CVC-ClinicDB_unet_r34) | 0.716 | 0.778 | 0.517 | 76.2 | – | – | – |
| unetpp-resnet34 (base_CVC-ClinicDB_unetpp_r34) | 0.710 | 0.771 | 0.530 | 75.3 | – | – | – |
| deeplabv3p-resnet50 (base_CVC-ColonDB_deeplabv3p_r50) | 0.597 | 0.676 | 0.344 | 114.4 | – | – | – |
| fpn-tu-pvt_v2_b2 (base_CVC-ColonDB_fpn_pvtv2b2) | 0.689 | 0.769 | 0.398 | 49.3 | – | – | – |
| segformer-mit_b2 (base_CVC-ColonDB_segformer_mitb2) | 0.698 | 0.777 | 0.406 | 49.6 | – | – | – |
| unet-resnet34 (base_CVC-ColonDB_unet_r34) | 0.605 | 0.676 | 0.387 | 121.9 | – | – | – |
| unetpp-resnet34 (base_CVC-ColonDB_unetpp_r34) | 0.583 | 0.651 | 0.366 | 139.5 | – | – | – |
| deeplabv3p-resnet50 (base_ETIS-LaribPolypDB_deeplabv3p_r50) | 0.600 | 0.674 | 0.434 | 109.9 | – | – | – |
| fpn-tu-pvt_v2_b2 (base_ETIS-LaribPolypDB_fpn_pvtv2b2) | 0.686 | 0.769 | 0.489 | 48.1 | – | – | – |
| segformer-mit_b2 (base_ETIS-LaribPolypDB_segformer_mitb2) | 0.670 | 0.748 | 0.482 | 59.0 | – | – | – |
| unet-resnet34 (base_ETIS-LaribPolypDB_unet_r34) | 0.617 | 0.684 | 0.476 | 100.3 | – | – | – |
| unetpp-resnet34 (base_ETIS-LaribPolypDB_unetpp_r34) | 0.631 | 0.694 | 0.489 | 111.5 | – | – | – |