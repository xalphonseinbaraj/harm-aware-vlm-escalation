# Revision analysis (theta=0.5, tau=0.5)

## Table A. Escalation policies (mean over n test images; IoU / Dice / Boundary-F1 / HD95 px; VLM calls; latency per image)

| Run | Policy | IoU | Dice | BF1 | HD95 | VLM calls | ms/image |
|---|---|---|---|---|---|---|---|
| light_s0__llama11b_p1_nim | never | 0.680 | 0.780 | 0.403 | 61.4 | 0/150 | 18 |
| light_s0__llama11b_p1_nim | always | 0.664 | 0.770 | 0.206 | 61.6 | 150/150 | 33686 |
| light_s0__llama11b_p1_nim | always+V | 0.668 | 0.773 | 0.242 | 61.8 | 150/150 | 33743 |
| light_s0__llama11b_p1_nim | conf-only | 0.673 | 0.775 | 0.356 | 62.0 | 47/150 | 10569 |
| light_s0__llama11b_p1_nim | gate | 0.678 | 0.778 | 0.376 | 62.0 | 47/150 | 10569 |
| light_s0__llama11b_p1_nim | gate+V | 0.678 | 0.779 | 0.378 | 61.9 | 47/150 | 10587 |
| light_s0__llama11b_p1_nim | oracle-diff | 0.680 | 0.780 | 0.363 | 61.7 | 64/150 | 14385 |
| light_s0__llama11b_p1_nim | oracle-gain | 0.693 | 0.791 | 0.381 | 59.9 | 57/150 | 12813 |
| light_s0__llama11b_p1_nim | random (same rate, 10k draws) | 0.675 [0.671, 0.679] | | | | 47/150 | |
| light_s0__qwen7b_p3 | never | 0.680 | 0.780 | 0.403 | 61.4 | 0/150 | 24 |
| light_s0__qwen7b_p3 | always | 0.674 | 0.775 | 0.375 | 60.9 | 150/150 | 10452 |
| light_s0__qwen7b_p3 | always+V | 0.674 | 0.775 | 0.375 | 60.9 | 150/150 | 10466 |
| light_s0__qwen7b_p3 | conf-only | 0.678 | 0.778 | 0.393 | 60.7 | 47/150 | 3293 |
| light_s0__qwen7b_p3 | gate | 0.678 | 0.777 | 0.392 | 60.7 | 47/150 | 3293 |
| light_s0__qwen7b_p3 | gate+V | 0.678 | 0.777 | 0.392 | 60.7 | 47/150 | 3298 |
| light_s0__qwen7b_p3 | oracle-diff | 0.676 | 0.776 | 0.390 | 60.9 | 64/150 | 4475 |
| light_s0__qwen7b_p3 | oracle-gain | 0.681 | 0.781 | 0.401 | 61.2 | 3/150 | 235 |
| light_s0__qwen7b_p3 | random (same rate, 10k draws) | 0.678 [0.677, 0.680] | | | | 47/150 | |
| light_s1__llama11b_p1_nim | never | 0.672 | 0.778 | 0.383 | 84.5 | 0/150 | 19 |
| light_s1__llama11b_p1_nim | always | 0.643 | 0.757 | 0.177 | 88.0 | 150/150 | 7267 |
| light_s1__llama11b_p1_nim | always+V | 0.649 | 0.761 | 0.223 | 88.1 | 150/150 | 7323 |
| light_s1__llama11b_p1_nim | conf-only | 0.658 | 0.767 | 0.307 | 87.2 | 70/150 | 3403 |
| light_s1__llama11b_p1_nim | gate | 0.659 | 0.766 | 0.314 | 85.6 | 70/150 | 3403 |
| light_s1__llama11b_p1_nim | gate+V | 0.661 | 0.769 | 0.325 | 85.2 | 70/150 | 3429 |
| light_s1__llama11b_p1_nim | oracle-diff | 0.662 | 0.768 | 0.310 | 85.5 | 71/150 | 3451 |
| light_s1__llama11b_p1_nim | oracle-gain | 0.680 | 0.784 | 0.358 | 84.4 | 36/150 | 1760 |
| light_s1__llama11b_p1_nim | random (same rate, 10k draws) | 0.659 [0.655, 0.663] | | | | 70/150 | |
| light_s1__qwen7b_p3 | never | 0.672 | 0.778 | 0.383 | 84.5 | 0/150 | 26 |
| light_s1__qwen7b_p3 | always | 0.672 | 0.777 | 0.369 | 83.7 | 150/150 | 10812 |
| light_s1__qwen7b_p3 | always+V | 0.672 | 0.777 | 0.369 | 83.7 | 150/150 | 10827 |
| light_s1__qwen7b_p3 | conf-only | 0.674 | 0.779 | 0.379 | 84.4 | 70/150 | 5061 |
| light_s1__qwen7b_p3 | gate | 0.673 | 0.778 | 0.377 | 84.1 | 70/150 | 5061 |
| light_s1__qwen7b_p3 | gate+V | 0.673 | 0.778 | 0.377 | 84.1 | 70/150 | 5068 |
| light_s1__qwen7b_p3 | oracle-diff | 0.671 | 0.777 | 0.374 | 84.0 | 71/150 | 5133 |
| light_s1__qwen7b_p3 | oracle-gain | 0.676 | 0.780 | 0.383 | 83.8 | 10/150 | 747 |
| light_s1__qwen7b_p3 | random (same rate, 10k draws) | 0.672 [0.670, 0.674] | | | | 70/150 | |
| light_s2__llama11b_p1_nim | never | 0.672 | 0.777 | 0.371 | 78.6 | 0/150 | 17 |
| light_s2__llama11b_p1_nim | always | 0.644 | 0.758 | 0.167 | 80.1 | 150/150 | 28184 |
| light_s2__llama11b_p1_nim | always+V | 0.653 | 0.764 | 0.238 | 79.9 | 150/150 | 28243 |
| light_s2__llama11b_p1_nim | conf-only | 0.657 | 0.766 | 0.292 | 79.1 | 73/150 | 13726 |
| light_s2__llama11b_p1_nim | gate | 0.663 | 0.770 | 0.306 | 79.0 | 73/150 | 13726 |
| light_s2__llama11b_p1_nim | gate+V | 0.665 | 0.771 | 0.317 | 79.0 | 73/150 | 13755 |
| light_s2__llama11b_p1_nim | oracle-diff | 0.668 | 0.773 | 0.308 | 78.9 | 71/150 | 13351 |
| light_s2__llama11b_p1_nim | oracle-gain | 0.684 | 0.787 | 0.337 | 77.7 | 46/150 | 8657 |
| light_s2__llama11b_p1_nim | random (same rate, 10k draws) | 0.658 [0.654, 0.663] | | | | 73/150 | |
| light_s2__qwen7b_p3 | never | 0.672 | 0.777 | 0.371 | 78.6 | 0/150 | 23 |
| light_s2__qwen7b_p3 | always | 0.667 | 0.772 | 0.355 | 77.2 | 150/150 | 10304 |
| light_s2__qwen7b_p3 | always+V | 0.667 | 0.772 | 0.355 | 77.2 | 150/150 | 10318 |
| light_s2__qwen7b_p3 | conf-only | 0.670 | 0.775 | 0.364 | 77.2 | 73/150 | 5028 |
| light_s2__qwen7b_p3 | gate | 0.668 | 0.773 | 0.359 | 77.8 | 73/150 | 5028 |
| light_s2__qwen7b_p3 | gate+V | 0.668 | 0.773 | 0.359 | 77.8 | 73/150 | 5035 |
| light_s2__qwen7b_p3 | oracle-diff | 0.667 | 0.772 | 0.360 | 78.7 | 71/150 | 4891 |
| light_s2__qwen7b_p3 | oracle-gain | 0.673 | 0.778 | 0.370 | 78.4 | 7/150 | 505 |
| light_s2__qwen7b_p3 | random (same rate, 10k draws) | 0.670 [0.668, 0.671] | | | | 73/150 | |
| segformer_s0__medgemma4b_p1 | never | 0.854 | 0.905 | 0.641 | 27.8 | 0/150 | 23 |
| segformer_s0__medgemma4b_p1 | always | 0.812 | 0.881 | 0.269 | 29.5 | 150/150 | 4891 |
| segformer_s0__medgemma4b_p1 | always+V | 0.824 | 0.888 | 0.374 | 29.1 | 150/150 | 4971 |
| segformer_s0__medgemma4b_p1 | conf-only | 0.848 | 0.901 | 0.602 | 28.0 | 27/150 | 901 |
| segformer_s0__medgemma4b_p1 | gate | 0.849 | 0.902 | 0.608 | 28.0 | 27/150 | 901 |
| segformer_s0__medgemma4b_p1 | gate+V | 0.850 | 0.902 | 0.612 | 28.0 | 27/150 | 916 |
| segformer_s0__medgemma4b_p1 | oracle-diff | 0.852 | 0.903 | 0.617 | 28.0 | 19/150 | 642 |
| segformer_s0__medgemma4b_p1 | oracle-gain | 0.855 | 0.906 | 0.629 | 27.6 | 22/150 | 739 |
| segformer_s0__medgemma4b_p1 | random (same rate, 10k draws) | 0.846 [0.844, 0.848] | | | | 27/150 | |
| segformer_s0__qwen7b_p3 | never | 0.854 | 0.905 | 0.641 | 27.8 | 0/150 | 31 |
| segformer_s0__qwen7b_p3 | always | 0.850 | 0.902 | 0.606 | 27.9 | 150/150 | 10460 |
| segformer_s0__qwen7b_p3 | always+V | 0.850 | 0.902 | 0.606 | 27.9 | 150/150 | 10475 |
| segformer_s0__qwen7b_p3 | conf-only | 0.853 | 0.904 | 0.641 | 27.9 | 27/150 | 1910 |
| segformer_s0__qwen7b_p3 | gate | 0.853 | 0.904 | 0.636 | 27.9 | 27/150 | 1910 |
| segformer_s0__qwen7b_p3 | gate+V | 0.853 | 0.904 | 0.636 | 27.9 | 27/150 | 1913 |
| segformer_s0__qwen7b_p3 | oracle-diff | 0.853 | 0.904 | 0.640 | 27.9 | 19/150 | 1354 |
| segformer_s0__qwen7b_p3 | oracle-gain | 0.854 | 0.905 | 0.643 | 27.9 | 2/150 | 172 |
| segformer_s0__qwen7b_p3 | random (same rate, 10k draws) | 0.853 [0.852, 0.854] | | | | 27/150 | |
| strong_s0__llama11b_p1_nim | never | 0.740 | 0.832 | 0.462 | 64.5 | 0/150 | 19 |
| strong_s0__llama11b_p1_nim | always | 0.702 | 0.807 | 0.187 | 66.4 | 150/150 | 35431 |
| strong_s0__llama11b_p1_nim | always+V | 0.709 | 0.811 | 0.236 | 66.7 | 150/150 | 35496 |
| strong_s0__llama11b_p1_nim | conf-only | 0.728 | 0.823 | 0.414 | 66.1 | 44/150 | 10409 |
| strong_s0__llama11b_p1_nim | gate | 0.729 | 0.823 | 0.416 | 66.0 | 44/150 | 10409 |
| strong_s0__llama11b_p1_nim | gate+V | 0.731 | 0.824 | 0.426 | 66.0 | 44/150 | 10428 |
| strong_s0__llama11b_p1_nim | oracle-diff | 0.733 | 0.825 | 0.403 | 65.0 | 52/150 | 12297 |
| strong_s0__llama11b_p1_nim | oracle-gain | 0.746 | 0.837 | 0.428 | 63.5 | 33/150 | 7812 |
| strong_s0__llama11b_p1_nim | random (same rate, 10k draws) | 0.729 [0.725, 0.733] | | | | 44/150 | |
| strong_s0__medgemma4b_p1 | never | 0.740 | 0.832 | 0.462 | 64.5 | 0/150 | 12 |
| strong_s0__medgemma4b_p1 | always | 0.704 | 0.808 | 0.190 | 66.0 | 150/150 | 4589 |
| strong_s0__medgemma4b_p1 | always+V | 0.709 | 0.812 | 0.234 | 66.7 | 150/150 | 4627 |
| strong_s0__medgemma4b_p1 | conf-only | 0.729 | 0.823 | 0.419 | 65.9 | 44/150 | 1356 |
| strong_s0__medgemma4b_p1 | gate | 0.732 | 0.824 | 0.425 | 65.7 | 44/150 | 1356 |
| strong_s0__medgemma4b_p1 | gate+V | 0.733 | 0.826 | 0.432 | 65.8 | 44/150 | 1367 |
| strong_s0__medgemma4b_p1 | oracle-diff | 0.734 | 0.826 | 0.413 | 64.9 | 52/150 | 1600 |
| strong_s0__medgemma4b_p1 | oracle-gain | 0.747 | 0.837 | 0.431 | 63.1 | 38/150 | 1173 |
| strong_s0__medgemma4b_p1 | random (same rate, 10k draws) | 0.729 [0.726, 0.733] | | | | 44/150 | |
| strong_s0__qwen7b_p1 | never | 0.740 | 0.832 | 0.462 | 64.5 | 0/150 | 23 |
| strong_s0__qwen7b_p1 | always | 0.734 | 0.827 | 0.413 | 64.6 | 150/150 | 9777 |
| strong_s0__qwen7b_p1 | always+V | 0.735 | 0.828 | 0.424 | 64.6 | 150/150 | 9794 |
| strong_s0__qwen7b_p1 | conf-only | 0.738 | 0.830 | 0.453 | 64.6 | 44/150 | 2886 |
| strong_s0__qwen7b_p1 | gate | 0.739 | 0.831 | 0.455 | 64.8 | 44/150 | 2886 |
| strong_s0__qwen7b_p1 | gate+V | 0.740 | 0.831 | 0.458 | 64.6 | 44/150 | 2891 |
| strong_s0__qwen7b_p1 | oracle-diff | 0.739 | 0.831 | 0.450 | 64.6 | 52/150 | 3406 |
| strong_s0__qwen7b_p1 | oracle-gain | 0.742 | 0.833 | 0.457 | 64.1 | 9/150 | 610 |
| strong_s0__qwen7b_p1 | random (same rate, 10k draws) | 0.738 [0.736, 0.740] | | | | 44/150 | |
| strong_s0__qwen7b_p3 | never | 0.740 | 0.832 | 0.462 | 64.5 | 0/150 | 25 |
| strong_s0__qwen7b_p3 | always | 0.737 | 0.829 | 0.441 | 64.3 | 150/150 | 10725 |
| strong_s0__qwen7b_p3 | always+V | 0.737 | 0.829 | 0.441 | 64.3 | 150/150 | 10740 |
| strong_s0__qwen7b_p3 | conf-only | 0.741 | 0.832 | 0.460 | 64.4 | 44/150 | 3165 |
| strong_s0__qwen7b_p3 | gate | 0.739 | 0.831 | 0.458 | 64.3 | 44/150 | 3165 |
| strong_s0__qwen7b_p3 | gate+V | 0.739 | 0.831 | 0.458 | 64.3 | 44/150 | 3170 |
| strong_s0__qwen7b_p3 | oracle-diff | 0.738 | 0.830 | 0.455 | 64.6 | 52/150 | 3736 |
| strong_s0__qwen7b_p3 | oracle-gain | 0.741 | 0.832 | 0.463 | 64.2 | 5/150 | 384 |
| strong_s0__qwen7b_p3 | random (same rate, 10k draws) | 0.739 [0.738, 0.740] | | | | 44/150 | |
| strong_s1__llama11b_p1_nim | never | 0.749 | 0.833 | 0.472 | 64.5 | 0/150 | 15 |
| strong_s1__llama11b_p1_nim | always | 0.723 | 0.817 | 0.242 | 65.4 | 150/150 | 21967 |
| strong_s1__llama11b_p1_nim | always+V | 0.726 | 0.819 | 0.267 | 65.7 | 150/150 | 22019 |
| strong_s1__llama11b_p1_nim | conf-only | 0.741 | 0.828 | 0.425 | 64.8 | 46/150 | 6749 |
| strong_s1__llama11b_p1_nim | gate | 0.742 | 0.828 | 0.423 | 64.8 | 46/150 | 6749 |
| strong_s1__llama11b_p1_nim | gate+V | 0.743 | 0.829 | 0.436 | 65.0 | 46/150 | 6765 |
| strong_s1__llama11b_p1_nim | oracle-diff | 0.745 | 0.830 | 0.429 | 64.8 | 42/150 | 6163 |
| strong_s1__llama11b_p1_nim | oracle-gain | 0.758 | 0.840 | 0.446 | 63.4 | 43/150 | 6310 |
| strong_s1__llama11b_p1_nim | random (same rate, 10k draws) | 0.741 [0.738, 0.745] | | | | 46/150 | |
| strong_s1__medgemma4b_p1 | never | 0.749 | 0.833 | 0.472 | 64.5 | 0/150 | 10 |
| strong_s1__medgemma4b_p1 | always | 0.721 | 0.815 | 0.239 | 64.6 | 150/150 | 4594 |
| strong_s1__medgemma4b_p1 | always+V | 0.725 | 0.818 | 0.262 | 64.8 | 150/150 | 4631 |
| strong_s1__medgemma4b_p1 | conf-only | 0.741 | 0.828 | 0.422 | 64.7 | 46/150 | 1418 |
| strong_s1__medgemma4b_p1 | gate | 0.741 | 0.827 | 0.422 | 64.5 | 46/150 | 1418 |
| strong_s1__medgemma4b_p1 | gate+V | 0.743 | 0.829 | 0.436 | 64.7 | 46/150 | 1429 |
| strong_s1__medgemma4b_p1 | oracle-diff | 0.744 | 0.829 | 0.428 | 64.9 | 42/150 | 1296 |
| strong_s1__medgemma4b_p1 | oracle-gain | 0.758 | 0.839 | 0.446 | 63.3 | 45/150 | 1387 |
| strong_s1__medgemma4b_p1 | random (same rate, 10k draws) | 0.741 [0.737, 0.744] | | | | 46/150 | |
| strong_s1__qwen7b_p3 | never | 0.749 | 0.833 | 0.472 | 64.5 | 0/150 | 26 |
| strong_s1__qwen7b_p3 | always | 0.743 | 0.829 | 0.435 | 64.3 | 150/150 | 10950 |
| strong_s1__qwen7b_p3 | always+V | 0.743 | 0.829 | 0.435 | 64.3 | 150/150 | 10964 |
| strong_s1__qwen7b_p3 | conf-only | 0.747 | 0.832 | 0.465 | 64.4 | 46/150 | 3378 |
| strong_s1__qwen7b_p3 | gate | 0.748 | 0.832 | 0.465 | 64.4 | 46/150 | 3378 |
| strong_s1__qwen7b_p3 | gate+V | 0.748 | 0.832 | 0.465 | 64.4 | 46/150 | 3382 |
| strong_s1__qwen7b_p3 | oracle-diff | 0.748 | 0.832 | 0.459 | 64.3 | 42/150 | 3087 |
| strong_s1__qwen7b_p3 | oracle-gain | 0.750 | 0.834 | 0.468 | 64.2 | 4/150 | 320 |
| strong_s1__qwen7b_p3 | random (same rate, 10k draws) | 0.747 [0.745, 0.749] | | | | 46/150 | |
| strong_s2__llama11b_p1_nim | never | 0.746 | 0.833 | 0.473 | 59.2 | 0/150 | 18 |
| strong_s2__llama11b_p1_nim | always | 0.722 | 0.819 | 0.211 | 61.0 | 150/150 | 10983 |
| strong_s2__llama11b_p1_nim | always+V | 0.723 | 0.820 | 0.234 | 61.2 | 150/150 | 11039 |
| strong_s2__llama11b_p1_nim | conf-only | 0.744 | 0.832 | 0.419 | 59.8 | 44/150 | 3236 |
| strong_s2__llama11b_p1_nim | gate | 0.746 | 0.833 | 0.437 | 60.5 | 44/150 | 3236 |
| strong_s2__llama11b_p1_nim | gate+V | 0.746 | 0.833 | 0.437 | 60.4 | 44/150 | 3252 |
| strong_s2__llama11b_p1_nim | oracle-diff | 0.748 | 0.835 | 0.429 | 59.9 | 47/150 | 3455 |
| strong_s2__llama11b_p1_nim | oracle-gain | 0.756 | 0.841 | 0.434 | 59.0 | 43/150 | 3163 |
| strong_s2__llama11b_p1_nim | random (same rate, 10k draws) | 0.739 [0.736, 0.743] | | | | 44/150 | |
| strong_s2__medgemma4b_p1 | never | 0.746 | 0.833 | 0.473 | 59.2 | 0/150 | 10 |
| strong_s2__medgemma4b_p1 | always | 0.722 | 0.819 | 0.220 | 60.3 | 150/150 | 4591 |
| strong_s2__medgemma4b_p1 | always+V | 0.723 | 0.820 | 0.241 | 61.0 | 150/150 | 4629 |
| strong_s2__medgemma4b_p1 | conf-only | 0.745 | 0.833 | 0.424 | 59.3 | 44/150 | 1356 |
| strong_s2__medgemma4b_p1 | gate | 0.745 | 0.833 | 0.434 | 60.0 | 44/150 | 1356 |
| strong_s2__medgemma4b_p1 | gate+V | 0.745 | 0.833 | 0.437 | 60.1 | 44/150 | 1367 |
| strong_s2__medgemma4b_p1 | oracle-diff | 0.748 | 0.834 | 0.432 | 59.5 | 47/150 | 1447 |
| strong_s2__medgemma4b_p1 | oracle-gain | 0.756 | 0.841 | 0.436 | 58.9 | 47/150 | 1447 |
| strong_s2__medgemma4b_p1 | random (same rate, 10k draws) | 0.739 [0.736, 0.743] | | | | 44/150 | |
| strong_s2__qwen7b_p3 | never | 0.746 | 0.833 | 0.473 | 59.2 | 0/150 | 25 |
| strong_s2__qwen7b_p3 | always | 0.742 | 0.829 | 0.454 | 59.2 | 150/150 | 10563 |
| strong_s2__qwen7b_p3 | always+V | 0.742 | 0.829 | 0.454 | 59.2 | 150/150 | 10575 |
| strong_s2__qwen7b_p3 | conf-only | 0.745 | 0.832 | 0.468 | 59.3 | 44/150 | 3118 |
| strong_s2__qwen7b_p3 | gate | 0.743 | 0.830 | 0.464 | 59.2 | 44/150 | 3118 |
| strong_s2__qwen7b_p3 | gate+V | 0.743 | 0.830 | 0.464 | 59.2 | 44/150 | 3121 |
| strong_s2__qwen7b_p3 | oracle-diff | 0.744 | 0.830 | 0.466 | 59.2 | 47/150 | 3329 |
| strong_s2__qwen7b_p3 | oracle-gain | 0.747 | 0.834 | 0.477 | 59.2 | 5/150 | 379 |
| strong_s2__qwen7b_p3 | random (same rate, 10k draws) | 0.745 [0.743, 0.746] | | | | 44/150 | |

## Table B. Paired Wilcoxon signed-rank tests on per-image IoU

| Run | gate+V vs always | gate+V vs always+V | gate vs always | gate+V vs never | gate+V vs gate | always vs never |
|---|---|---|---|---|---|---|
| light_s0__llama11b_p1_nim | <0.0001 | 0.0008 | <0.0001 | 0.6291 | 1.0000 | 0.0002 |
| light_s0__qwen7b_p3 | 0.0022 | 0.0022 | 0.0022 | 0.0609 | 1.0000 | 0.0005 |
| light_s1__llama11b_p1_nim | <0.0001 | <0.0001 | <0.0001 | 0.0005 | 0.1095 | <0.0001 |
| light_s1__qwen7b_p3 | 0.2393 | 0.2393 | 0.2393 | 0.7537 | 1.0000 | 0.6071 |
| light_s2__llama11b_p1_nim | <0.0001 | <0.0001 | <0.0001 | 0.0720 | 0.2411 | <0.0001 |
| light_s2__qwen7b_p3 | 0.5337 | 0.5337 | 0.5337 | 0.0038 | 1.0000 | 0.0046 |
| segformer_s0__medgemma4b_p1 | <0.0001 | <0.0001 | <0.0001 | 0.0106 | 0.3454 | <0.0001 |
| segformer_s0__qwen7b_p3 | 0.0051 | 0.0051 | 0.0051 | 0.1159 | 1.0000 | 0.0008 |
| strong_s0__llama11b_p1_nim | <0.0001 | <0.0001 | <0.0001 | 0.0019 | 0.2626 | <0.0001 |
| strong_s0__medgemma4b_p1 | <0.0001 | <0.0001 | <0.0001 | 0.0079 | 0.3270 | <0.0001 |
| strong_s0__qwen7b_p1 | 0.0011 | 0.0025 | 0.0024 | 0.6858 | 0.2850 | 0.0034 |
| strong_s0__qwen7b_p3 | 0.0480 | 0.0480 | 0.0480 | 0.3454 | 1.0000 | 0.0366 |
| strong_s1__llama11b_p1_nim | <0.0001 | <0.0001 | <0.0001 | 0.0503 | 0.2026 | <0.0001 |
| strong_s1__medgemma4b_p1 | <0.0001 | <0.0001 | <0.0001 | 0.0684 | 0.1688 | <0.0001 |
| strong_s1__qwen7b_p3 | 0.0056 | 0.0056 | 0.0056 | 0.1763 | 1.0000 | 0.0015 |
| strong_s2__llama11b_p1_nim | <0.0001 | <0.0001 | <0.0001 | 0.6967 | 0.3173 | <0.0001 |
| strong_s2__medgemma4b_p1 | <0.0001 | <0.0001 | <0.0001 | 0.5907 | 1.0000 | <0.0001 |
| strong_s2__qwen7b_p3 | 0.1095 | 0.1095 | 0.1095 | 0.0367 | 1.0000 | 0.0087 |

## Table C. Gate vs. true difficulty (backbone IoU < 0.7)  [Reviewer 3, Results 1]

| Run | esc. rate | hard rate | Gate P | Gate R | Gate F1 | Conf-only P / R / F1 | Random P (95% CI) | AUROC gate | AUROC conf | P(random ≥ gate IoU) |
|---|---|---|---|---|---|---|---|---|---|---|
| light_s0__llama11b_p1_nim | 0.313 | 0.427 | 0.830 | 0.609 | 0.703 | 0.596 / 0.438 / 0.505 | 0.427 (0.319–0.532) | 0.880 | 0.672 | 0.074 |
| light_s0__qwen7b_p3 | 0.313 | 0.427 | 0.830 | 0.609 | 0.703 | 0.596 / 0.438 / 0.505 | 0.427 (0.298–0.553) | 0.880 | 0.672 | 0.784 |
| light_s1__llama11b_p1_nim | 0.467 | 0.473 | 0.757 | 0.746 | 0.752 | 0.671 / 0.662 / 0.667 | 0.473 (0.386–0.557) | 0.858 | 0.703 | 0.501 |
| light_s1__qwen7b_p3 | 0.467 | 0.473 | 0.757 | 0.746 | 0.752 | 0.671 / 0.662 / 0.667 | 0.473 (0.386–0.557) | 0.858 | 0.703 | 0.189 |
| light_s2__llama11b_p1_nim | 0.487 | 0.473 | 0.753 | 0.775 | 0.764 | 0.562 / 0.577 / 0.569 | 0.473 (0.397–0.548) | 0.863 | 0.635 | 0.026 |
| light_s2__qwen7b_p3 | 0.487 | 0.473 | 0.753 | 0.775 | 0.764 | 0.562 / 0.577 / 0.569 | 0.473 (0.397–0.562) | 0.863 | 0.635 | 0.977 |
| segformer_s0__medgemma4b_p1 | 0.180 | 0.127 | 0.333 | 0.474 | 0.391 | 0.444 / 0.632 / 0.522 | 0.125 (0.037–0.259) | 0.684 | 0.773 | 0.001 |
| segformer_s0__qwen7b_p3 | 0.180 | 0.127 | 0.333 | 0.474 | 0.391 | 0.444 / 0.632 / 0.522 | 0.127 (0.037–0.259) | 0.684 | 0.773 | 0.693 |
| strong_s0__llama11b_p1_nim | 0.293 | 0.347 | 0.705 | 0.596 | 0.646 | 0.591 / 0.500 / 0.542 | 0.346 (0.227–0.455) | 0.881 | 0.744 | 0.398 |
| strong_s0__medgemma4b_p1 | 0.293 | 0.347 | 0.705 | 0.596 | 0.646 | 0.591 / 0.500 / 0.542 | 0.346 (0.227–0.455) | 0.881 | 0.744 | 0.118 |
| strong_s0__qwen7b_p1 | 0.293 | 0.347 | 0.705 | 0.596 | 0.646 | 0.591 / 0.500 / 0.542 | 0.347 (0.227–0.455) | 0.881 | 0.744 | 0.352 |
| strong_s0__qwen7b_p3 | 0.293 | 0.347 | 0.705 | 0.596 | 0.646 | 0.591 / 0.500 / 0.542 | 0.347 (0.227–0.455) | 0.881 | 0.744 | 0.632 |
| strong_s1__llama11b_p1_nim | 0.307 | 0.280 | 0.674 | 0.738 | 0.705 | 0.391 / 0.429 / 0.409 | 0.280 (0.174–0.391) | 0.888 | 0.710 | 0.406 |
| strong_s1__medgemma4b_p1 | 0.307 | 0.280 | 0.674 | 0.738 | 0.705 | 0.391 / 0.429 / 0.409 | 0.281 (0.174–0.391) | 0.888 | 0.710 | 0.352 |
| strong_s1__qwen7b_p3 | 0.307 | 0.280 | 0.674 | 0.738 | 0.705 | 0.391 / 0.429 / 0.409 | 0.280 (0.174–0.391) | 0.888 | 0.710 | 0.314 |
| strong_s2__llama11b_p1_nim | 0.293 | 0.313 | 0.705 | 0.660 | 0.681 | 0.523 / 0.489 / 0.505 | 0.313 (0.205–0.432) | 0.874 | 0.722 | 0.000 |
| strong_s2__medgemma4b_p1 | 0.293 | 0.313 | 0.705 | 0.660 | 0.681 | 0.523 / 0.489 / 0.505 | 0.314 (0.205–0.432) | 0.874 | 0.722 | 0.000 |
| strong_s2__qwen7b_p3 | 0.293 | 0.313 | 0.705 | 0.660 | 0.681 | 0.523 / 0.489 / 0.505 | 0.315 (0.205–0.432) | 0.874 | 0.722 | 0.973 |

## Table D. Confidence-head calibration  [Reviewer 3, Related work 1]

| Run | ECE | MAE | Pearson r | Spearman ρ | conf range |
|---|---|---|---|---|---|
| light_s0__llama11b_p1_nim | 0.011 | 0.191 | 0.387 | 0.380 | 0.460–0.820 |
| light_s0__qwen7b_p3 | 0.011 | 0.191 | 0.387 | 0.380 | 0.460–0.820 |
| light_s1__llama11b_p1_nim | 0.053 | 0.172 | 0.431 | 0.438 | 0.329–0.811 |
| light_s1__qwen7b_p3 | 0.053 | 0.172 | 0.431 | 0.438 | 0.329–0.811 |
| light_s2__llama11b_p1_nim | 0.030 | 0.175 | 0.250 | 0.283 | 0.517–0.831 |
| light_s2__qwen7b_p3 | 0.030 | 0.175 | 0.250 | 0.283 | 0.517–0.831 |
| segformer_s0__medgemma4b_p1 | 0.124 | 0.124 | 0.455 | 0.769 | 0.883–0.991 |
| segformer_s0__qwen7b_p3 | 0.124 | 0.124 | 0.455 | 0.769 | 0.883–0.991 |
| strong_s0__llama11b_p1_nim | 0.026 | 0.138 | 0.433 | 0.535 | 0.544–0.899 |
| strong_s0__medgemma4b_p1 | 0.026 | 0.138 | 0.433 | 0.535 | 0.544–0.899 |
| strong_s0__qwen7b_p1 | 0.026 | 0.138 | 0.433 | 0.535 | 0.544–0.899 |
| strong_s0__qwen7b_p3 | 0.026 | 0.138 | 0.433 | 0.535 | 0.544–0.899 |
| strong_s1__llama11b_p1_nim | 0.018 | 0.148 | 0.438 | 0.428 | 0.458–0.887 |
| strong_s1__medgemma4b_p1 | 0.018 | 0.148 | 0.438 | 0.428 | 0.458–0.887 |
| strong_s1__qwen7b_p3 | 0.018 | 0.148 | 0.438 | 0.428 | 0.458–0.887 |
| strong_s2__llama11b_p1_nim | 0.053 | 0.173 | 0.327 | 0.358 | 0.405–0.859 |
| strong_s2__medgemma4b_p1 | 0.053 | 0.173 | 0.327 | 0.358 | 0.405–0.859 |
| strong_s2__qwen7b_p3 | 0.053 | 0.173 | 0.327 | 0.358 | 0.405–0.859 |

## Table E. Why indiscriminate escalation hurts: effect of the VLM edit by difficulty, and VLM behaviour

| Run | ΔIoU easy | % easy hurt | ΔIoU hard | % hard helped | % hard hurt | adjustments (all images) | template-copy rate | verif. accept rate | F vs helped (r, p) |
|---|---|---|---|---|---|---|---|---|---|
| light_s0__llama11b_p1_nim | -0.029 | 69% | -0.000 | 53% | 39% | {'expand': 142, 'no_change': 8} | 0% | 0.879 | 0.177, 0.0357 |
| light_s0__qwen7b_p3 | -0.004 | 9% | -0.009 | 2% | 25% | {'no_change': 123, 'shrink': 27} | 82% | 1.000 | -0.279, 0.1585 |
| light_s1__llama11b_p1_nim | -0.036 | 75% | -0.022 | 34% | 61% | {'expand': 132, 'no_change': 12, 'shift_boundary': 5, 'shrink': 1} | 0% | 0.812 | 0.038, 0.6540 |
| light_s1__qwen7b_p3 | 0.001 | 5% | -0.002 | 8% | 14% | {'no_change': 126, 'shrink': 24} | 84% | 1.000 | -0.061, 0.7772 |
| light_s2__llama11b_p1_nim | -0.045 | 76% | -0.009 | 49% | 46% | {'expand': 137, 'no_change': 10, 'shift_boundary': 3} | 0% | 0.757 | 0.034, 0.6911 |
| light_s2__qwen7b_p3 | 0.001 | 4% | -0.011 | 1% | 24% | {'no_change': 122, 'shrink': 28} | 81% | 1.000 | 0.220, 0.2601 |
| segformer_s0__medgemma4b_p1 | -0.046 | 86% | -0.013 | 37% | 63% | {'expand': 136, 'shift_boundary': 11, 'shrink': 3} | 0% | 0.673 | 0.001, 0.9923 |
| segformer_s0__qwen7b_p3 | -0.004 | 9% | -0.003 | 5% | 11% | {'no_change': 134, 'shrink': 16} | 89% | 1.000 | 0.287, 0.2814 |
| strong_s0__llama11b_p1_nim | -0.047 | 79% | -0.021 | 37% | 60% | {'expand': 140, 'no_change': 9, 'shift_boundary': 1} | 0% | 0.837 | 0.037, 0.6615 |
| strong_s0__medgemma4b_p1 | -0.047 | 83% | -0.017 | 40% | 60% | {'expand': 137, 'shift_boundary': 10, 'shrink': 3} | 0% | 0.833 | -0.020, 0.8109 |
| strong_s0__qwen7b_p1 | -0.008 | 14% | -0.003 | 6% | 13% | {'no_change': 120, 'expand': 27, 'shrink': 2, 'shift_boundary': 1} | 0% | 0.767 | 0.070, 0.7120 |
| strong_s0__qwen7b_p3 | -0.002 | 8% | -0.006 | 0% | 13% | {'no_change': 130, 'shrink': 20} | 87% | 1.000 | 0.001, 0.9970 |
| strong_s1__llama11b_p1_nim | -0.031 | 72% | -0.013 | 48% | 48% | {'expand': 141, 'no_change': 8, 'shift_boundary': 1} | 0% | 0.887 | 0.056, 0.5074 |
| strong_s1__medgemma4b_p1 | -0.032 | 75% | -0.016 | 43% | 55% | {'expand': 137, 'shift_boundary': 9, 'shrink': 4} | 0% | 0.886 | -0.004, 0.9583 |
| strong_s1__qwen7b_p3 | -0.007 | 12% | -0.004 | 7% | 17% | {'no_change': 126, 'shrink': 24} | 84% | 1.000 | -0.176, 0.4119 |
| strong_s2__llama11b_p1_nim | -0.038 | 78% | 0.006 | 57% | 38% | {'expand': 141, 'no_change': 8, 'shift_boundary': 1} | 0% | 0.922 | -0.027, 0.7537 |
| strong_s2__medgemma4b_p1 | -0.037 | 81% | 0.004 | 57% | 40% | {'expand': 134, 'shift_boundary': 14, 'shrink': 2} | 0% | 0.906 | 0.008, 0.9263 |
| strong_s2__qwen7b_p3 | -0.003 | 7% | -0.009 | 2% | 19% | {'no_change': 129, 'shrink': 21} | 86% | 1.000 | -0.123, 0.5961 |

## Table F. Timing breakdown (ms, mean per image)

| Run | backbone | gate | overlay | VLM call | verification |
|---|---|---|---|---|---|
| light_s0__llama11b_p1_nim | 17.6 | 2.69 | 99.2 | 33567 | 56.6 |
| light_s0__qwen7b_p3 | 23.8 | 2.72 | 105.2 | 10320 | 14.7 |
| light_s1__llama11b_p1_nim | 19.1 | 2.48 | 111.7 | 7134 | 56.2 |
| light_s1__qwen7b_p3 | 25.9 | 2.65 | 117.6 | 10666 | 14.4 |
| light_s2__llama11b_p1_nim | 17.2 | 2.51 | 108.0 | 28057 | 59.2 |
| light_s2__qwen7b_p3 | 22.7 | 2.82 | 112.7 | 10166 | 14.2 |
| segformer_s0__medgemma4b_p1 | 23.5 | 2.02 | 93.7 | 4771 | 80.7 |
| segformer_s0__qwen7b_p3 | 30.8 | 2.34 | 99.4 | 10327 | 15.1 |
| strong_s0__llama11b_p1_nim | 19.3 | 2.99 | 78.8 | 35330 | 65.2 |
| strong_s0__medgemma4b_p1 | 11.9 | 2.56 | 97.3 | 4478 | 37.4 |
| strong_s0__qwen7b_p1 | 22.6 | 2.60 | 104.1 | 9648 | 16.9 |
| strong_s0__qwen7b_p3 | 25.0 | 2.60 | 113.6 | 10584 | 14.9 |
| strong_s1__llama11b_p1_nim | 15.0 | 2.60 | 112.0 | 21838 | 51.6 |
| strong_s1__medgemma4b_p1 | 10.3 | 2.84 | 114.5 | 4466 | 37.1 |
| strong_s1__qwen7b_p3 | 26.4 | 2.40 | 104.4 | 10817 | 14.2 |
| strong_s2__llama11b_p1_nim | 17.8 | 2.62 | 98.2 | 10865 | 55.3 |
| strong_s2__medgemma4b_p1 | 10.2 | 2.62 | 107.8 | 4471 | 37.1 |
| strong_s2__qwen7b_p3 | 24.9 | 2.67 | 108.2 | 10427 | 12.0 |

## Table G. Pooled across seeds (mean ± SD of per-seed means; pooled Wilcoxon over all image-seed pairs)

| Backbone / VLM | seeds | never | always | gate+V (proposed) | esc. rate | gate+V vs always p | gate+V vs never p |
|---|---|---|---|---|---|---|---|
| light / llama11b_p1_nim | 3 | 0.675 ± 0.005 | 0.650 ± 0.012 | 0.668 ± 0.009 | 0.422 | <0.0001 | 0.0005 |
| light / qwen7b_p3 | 3 | 0.675 ± 0.005 | 0.671 ± 0.003 | 0.673 ± 0.005 | 0.422 | 0.0029 | 0.0076 |
| segformer / medgemma4b_p1 | 1 | 0.854 ± 0.000 | 0.812 ± 0.000 | 0.850 ± 0.000 | 0.180 | <0.0001 | 0.0106 |
| segformer / qwen7b_p3 | 1 | 0.854 ± 0.000 | 0.850 ± 0.000 | 0.853 ± 0.000 | 0.180 | 0.0051 | 0.1159 |
| strong / llama11b_p1_nim | 3 | 0.745 ± 0.005 | 0.716 ± 0.012 | 0.740 ± 0.008 | 0.298 | <0.0001 | 0.0009 |
| strong / medgemma4b_p1 | 3 | 0.745 ± 0.005 | 0.716 ± 0.011 | 0.740 ± 0.007 | 0.298 | <0.0001 | 0.0033 |
| strong / qwen7b_p1 | 1 | 0.740 ± 0.000 | 0.734 ± 0.000 | 0.740 ± 0.000 | 0.293 | 0.0011 | 0.6858 |
| strong / qwen7b_p3 | 3 | 0.745 ± 0.005 | 0.740 ± 0.003 | 0.743 ± 0.004 | 0.298 | 0.0001 | 0.0081 |

## Table H. Matched baselines (same split / resolution / augmentation / model selection)  [Reviewer 1, c6]

| Model | IoU | Dice | BF1 | HD95 | Params (M) | GMACs | ms/img |
|---|---|---|---|---|---|---|---|
| deeplabv3p-resnet50 (base_deeplabv3p_r50) | 0.835 | 0.896 | 0.601 | 36.1 | 26.68 | 20.57 | 7.6 |
| fpn-tu-pvt_v2_b2 (base_fpn_pvtv2b2) | 0.853 | 0.906 | 0.646 | 26.7 | 26.74 | 17.31 | 15.4 |
| segformer-mit_b2 (base_segformer_mitb2) | 0.854 | 0.905 | 0.641 | 27.8 | 24.72 | 13.14 | 11.6 |
| unet-resnet34 (base_unet_r34) | 0.841 | 0.900 | 0.633 | 30.9 | 24.44 | 17.58 | 7.1 |
| unetpp-resnet34 (base_unetpp_r34) | 0.854 | 0.908 | 0.646 | 27.3 | 26.08 | 41.36 | 15.6 |

## Table I. Sensitivity to the confidence-loss weight λ (Eq. 1), seed 0, strong augmentation  [Reviewer 3, Method 2]

| λ | test IoU | test Dice | conf–IoU Pearson r | ECE | gate AUROC (val) |
|---|---|---|---|---|---|
| 0.1 | 0.743 | 0.833 | 0.270 | 0.018 | 0.874 |
| 0.5 | 0.740 | 0.832 | 0.433 | 0.026 | 0.868 |
| 1.0 | 0.744 | 0.828 | 0.521 | 0.053 | 0.908 |