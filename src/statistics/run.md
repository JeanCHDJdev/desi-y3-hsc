## run

-------------

This markdown documents some of the run commands that are frequently used within the cross correlation
pipeline :

python run_corr.py -j -k -s 5 -c 4 -t1 ELGnotqso -w PIP -r1 5 -o outdir

python run_corr.py -j -k -o auto/auto_allsky_ELG_PIP_rp_2 -s 0 -c 4 -t1 ELGnotqso -w PIP -r1 5 -d rp

python run_corr.py -o auto/auto_allsky_ELG_PIP_rp_nojk -s 0 -c 200 -t1 ELGnotqso -w PIP -r1 5 -d rp -k



## Making Results

-----------------

#### DESI autocorrelations

--------------------------

Number of random files used for DESI : 5
* ELGnotqso :
    - bins : `np.arange(0.8, 1.625, 0.025)`
    - python run_corr.py -o results/autos_j64_ns256_NGC -s 0 -c 200 -t1 ELGnotqso -w PIP -k
* LRG :
    - bins : `np.arange(0.4, 1.125, 0.025)`
    - python run_corr.py -o results/autos_j64_ns256_NGC -s 0 -c 200 -t1 LRG -w PIP -k
* QSO:
   - bins : `np.arange(0.9, 2.95, 0.15)`
   - python run_corr.py -o results/autos_j64_ns256_NGC -s 0 -c 200 -t1 QSO -w PIP -k
* BGS:
   - bins : `np.arange(0, 0.525, 0.025)`
   - python run_corr.py -o results/autos_j64_ns256_NGC -s 0 -c 200 -t1 BGS_ANY -w PIP -k
Commands
```bash
setcc && python run_corr.py -o outputs/nonKP_FKP_davis_peebles/autos_j64_ns256_NGC -s 0 -c 200 -t1 ELGnotqso -w nonKP -k -ns 64 -re 256 -a 1 && python run_corr.py -o outputs/nonKP_FKP_davis_peebles/autos_j64_ns256_NGC -s 0 -c 200 -t1 LRG -w nonKP -k -ns 64 -re 256 -a 1 && python run_corr.py -o outputs/nonKP_FKP_davis_peebles/autos_j64_ns256_NGC -s 0 -c 200 -t1 BGS_ANY -w nonKP -k -ns 64 -re 256 -a 1 && python run_corr.py -o outputs/nonKP_FKP_davis_peebles/autos_j64_ns256_NGC -s 0 -c 200 -t1 QSO -w nonKP -k -ns 64 -re 256 -a 1 
```

```bash
setcc && python run_corr.py -o results/autos_j64_ns256_NGC -s 0 -c 200 -t1 QSO -w PIP -k -ns 64 -re 256 -a 1 && python run_corr.py -o results/autos_j64_ns256_NGC -s 0 -c 200 -t1 BGS_ANY -w PIP -k -ns 64 -re 256 -a 1 -j
```

setcc && python run_corr.py -o outputs/nonKP_FKP_landyszalay_0_05/cross_j64_ns256_zbin -s 0 -c 200 -t1 LRG -t2 HSC -w nonKP -a 1 -ns 64 -re 256 -z

setcc && python run_corr.py -o outputs/results_sims_rcc_v2_0_1/autos_j64_ns256_HSC -s 5 -c 200 -t1 HSC -w nonKP -a 1 -ns 64 -re 256 -z -r1 6 -r2 6

setcc && python run_corr.py -o outputs/results_sims_rcc_v2_0_1/autos_j64_ns256_HSC -s 5 -c 200 -t1 HSC -w nonKP -a 1 -ns 64 -re 256 -z -r1 6 -r2 6

#### HSC autocorrelation settings

---------------------------------

* HSC:
   - bins : `np.arange(0.3, 1.8, 0.3)`
   - python run_corr.py -o results/autos_j64_ns256_NGC -s 0 -c 200 -t1 BGS_ANY -w PIP -k
For HSC compute time, we don't skip MOC. We downsample randoms by 5 (density : 20 gal/arcmin)
Commands
```bash
setcc && python run_corr.py -o outputs/results/autos_j64_ns256 -s 0 -c 200 -t1 HSC -ns 64 -re 256 -r2 5 -j
```
with z-bins : 
```bash
setcc && python run_corr.py -o outputs/results/autos_j64_ns256_zbin -s 0 -c 200 -t1 HSC -ns 64 -re 256 -r1 5 -r2 5 -z -j
```
with mini-bins :
NOTE : not excluding "calibration cut" which is bad for first 2 bins D:
```bash
setcc && python run_corr.py -o outputs/results_hsc/minibins_j64_ns256 -s 0 -c 200 -t1 HSC -ns 64 -re 256 -r1 40 -r2 40 -z
```

setcc && python run_corr.py -o outputs/results_sims/autos_j64_ns256_zbin -s 5 -c 200 -t1 LRG -ns 64 -re 256 -k -a 1 && python run_corr.py -o outputs/results_sims/autos_j64_ns256_zbin -s 5 -c 200 -t1 BGS_ANY -ns 64 -re 256 -k -a 1 && python run_corr.py -o outputs/results_sims/autos_j64_ns256_zbin -s 5 -c 200 -t1 ELGnotqso -ns 64 -re 256 -k -a 1 && python run_corr.py -o outputs/results_sims/autos_j64_ns256_zbin -s 5 -c 200 -t1 HSC -ns 64 -re 256

setcc && python run_corr.py -o outputs/results_sims/cross_j64_ns256_zbin -s 5 -c 200 -t1 ELGnotqso -t2 HSC -ns 64 -re 256 -r2 4 -w PIP && python run_corr.py -o outputs/results/cross_j64_ns256_zbin -s 5 -c 200 -t1 LRG -t2 HSC -ns 64 -re 256 -r2 4 -w PIP && python run_corr.py -o outputs/results/cross_j64_ns256_zbin -s 5 -c 200 -t1 BGS_ANY -t2 HSC -ns 64 -re 256 -r2 4 -w PIP -j

##### HSC-DESI cross-correlation

--------------------------------

Same bins as previous settings. 
Command :
```bash
setcc && python run_corr.py -o outputs/results_2/cross_j64_ns256_zbin -s 0 -c 200 -t1 ELGnotqso -t2 HSC -ns 64 -re 256 -r2 15 -r2 2 -z -w PIP -j && python run_corr.py -o outputs/results/cross_j64_ns256_zbin -s 0 -c 200 -t1 LRG -t2 HSC -ns 64 -re 256 -r2 15 -r2 2 -z -w PIP -j
```

```bash
setcc && python run_corr.py -o outputs/results/cross_j64_ns256_zbin -s 0 -c 200 -t1 BGS_ANY -t2 HSC -ns 64 -re 256 -r2 15 -r2 2 -z -w PIP -j && python run_corr.py -o outputs/results/cross_j64_ns256_zbin -s 0 -c 200 -t1 QSO -t2 HSC -ns 64 -re 256 -r2 15 -r2 2 -z -w PIP -j
```

## HSC-DESI cross correlation nonKP FKP davis peebles
```bash
setcc && python run_corr.py -o outputs/nonKP_FKP_davis_peebles/cross_j64_ns256_zbin -s 0 -c 200 -t1 BGS_ANY -t2 HSC -ns 64 -re 256 -r1 4 -r2 15 -z -w nonKP && python run_corr.py -o outputs/nonKP_FKP_davis_peebles/cross_j64_ns256_zbin -s 0 -c 200 -t1 QSO -t2 HSC -ns 64 -re 256 -r1 4 -r2 15 -z -w nonKP && python run_corr.py -o outputs/nonKP_FKP_davis_peebles/cross_j64_ns256_zbin -s 0 -c 200 -t1 ELGnotqso -t2 HSC -ns 64 -re 256 -r1 4 -r2 15 -z -w nonKP && python run_corr.py -o outputs/nonKP_FKP_davis_peebles/cross_j64_ns256_zbin -s 0 -c 200 -t1 LRG -t2 HSC -ns 64 -re 256 -r1 4 -r2 15 -z -w nonKP
```

# 17/05 morning
setcc && python run_corr.py -o outputs/nonKP_FKP_davispeebles_0_05/cross_j64_ns256_zbin -s 0 -c 200 -t1 ELGnotqso -t2 HSC -ns 64 -re 256 -r2 15 -z -w nonKP && python run_corr.py -o outputs/nonKP_FKP_davispeebles_0_05/cross_j64_ns256_zbin -s 0 -c 200 -t1 LRG -t2 HSC -ns 64 -re 256 -r2 15 -z -w nonKP && python run_corr.py -o outputs/nonKP_FKP_davispeebles_0_05/cross_j64_ns256_zbin -s 0 -c 200 -t1 BGS_ANY -t2 HSC -ns 64 -re 256 -r2 15 -z -w nonKP && python run_corr.py -o outputs/nonKP_FKP_davispeebles_0_05/cross_j64_ns256_zbin -s 0 -c 200 -t1 QSO -t2 HSC -ns 64 -re 256 -r2 15 -z -w nonKP 

setcc && python run_corr.py -o outputs/nonKP_FKP_davispeebles_0_05/autos_j64_ns256_NGC -s 0 -c 200 -t1 LRG -ns 64 -re 256 -z -k -a 1 -w nonKP && python run_corr.py -o outputs/nonKP_FKP_davispeebles_0_05/autos_j64_ns256_NGC -s 0 -c 200 -t1 BGS_ANY -ns 64 -re 256 -r1 40 -r2 40 -z -k -a 1 -w nonKP && python run_corr.py -o outputs/nonKP_FKP_davispeebles_0_05/autos_j64_ns256_NGC -s 0 -c 200 -t1 ELGnotqso -ns 64 -re 256 -r1 40 -r2 40 -z -k -a 1 -w nonKP && python run_corr.py -o outputs/nonKP_FKP_davispeebles_0_05/autos_j64_ns256_NGC -s 0 -c 200 -t1 QSO -ns 64 -re 256 -r1 40 -r2 40 -z -k -a 1 -w nonKP 


# TODO : compute r_cc based on simulations

setcc && python run_corr.py -o outputs/results_sims_rcc_v2_0_1/cross_j64_ns256_zbin -s 5 -c 200 -t1 BGS_ANY -t2 HSC -ns 64 -re 256 -r1 4 -r2 4 && python run_corr.py -o outputs/results_sims_rcc_v2_0_1/cross_j64_ns256_zbin -s 5 -c 200 -t1 LRG -t2 HSC -ns 64 -re 256 -r1 4 -r2 4

setcc && python run_corr.py -o outputs/results_sims_rcc_v2_0_1/autos_j64_ns256_zbin -s 5 -c 200 -t1 HSC -ns 64 -re 256 -r1 4 -r2 4 && python run_corr.py -o outputs/results_sims_rcc_v2_0_1/autos_j64_ns256_zbin -s 5 -c 200 -t1 LRG -ns 64 -re 256 -r1 4 -r2 4 && python run_corr.py -o outputs/results_sims_rcc_v2_0_1/autos_j64_ns256_zbin -s 5 -c 200 -t1 BGS_ANY -ns 64 -re 256 -r1 4 -r2 4

setcc && python run_corr.py -o outputs/results_sims_rcc_v2_0_1/autos_j64_ns256_zbin -s 5 -c 200 -t1 ELGnotqso -t2 LRG -ns 64 -re 256 -r1 6 -r2 6 -k -a 1




### Commands to run

---------------------

### Real Data
- Cross correlations :
OUTDIR="outputs/current4_offset/cross_j64_ns256" && setcc && python run_corr.py -o $OUTDIR -s 0 -c 200 -t1 ELGnotqso -t2 HSC -ns 64 -re 256 -r1 5 -r2 50 -z -j && python run_corr.py -o $OUTDIR -s 0 -c 200 -t1 LRG -t2 HSC -ns 64 -re 256 -r1 5 -r2 50 -z -j && python run_corr.py -o $OUTDIR -s 0 -c 200 -t1 BGS_ANY -t2 HSC -ns 64 -re 256 -r1 5 -r2 50 -z -j && python run_corr.py -o $OUTDIR -s 0 -c 200 -t1 QSO -t2 HSC -ns 64 -re 256 -r1 5 -r2 50 -z -j

- Auto correlation :
OUTDIR="outputs/current4_offset/auto_j64_ns256_NGC" && setcc && python run_corr.py -o $OUTDIR -s 0 -c 200 -t1 ELGnotqso -ns 64 -re 256 -r1 5 -r2 50 -z -j -k -a 1 && python run_corr.py -o $OUTDIR -s 0 -c 200 -t1 LRG -ns 64 -re 256 -r1 5 -r2 50 -z -j -k -a 1 && python run_corr.py -o $OUTDIR -s 0 -c 200 -t1 BGS_ANY -ns 64 -re 256 -r1 5 -r2 50 -z -j -k -a 1 && python run_corr.py -o $OUTDIR -s 0 -c 200 -t1 QSO -ns 64 -re 256 -r1 5 -r2 50 -z -j -k -a 1 && python run_corr.py -o outputs/current4/auto_j64_ns256_HSC -s 0 -c 200 -t1 HSC -ns 64 -re 256 -r1 50 -r2 50 -z -j

### Simulations :
- Autos :
OUTDIR="outputs/sims2/auto_j64_ns256" && setcc && python run_corr.py -o $OUTDIR -s 5 -c 200 -t1 ELGnotqso -ns 64 -re 256 -r1 3 -r2 3 -k -a 1 && python run_corr.py -o $OUTDIR -s 5 -c 200 -t1 LRG -ns 64 -re 256 -r1 3 -r2 3 -k -a 1 && python run_corr.py -o $OUTDIR -s 5 -c 200 -t1 BGS_ANY -ns 64 -re 256 -r1 3 -r2 3 -k -a 1 && python run_corr.py -o $OUTDIR -s 5 -c 200 -t1 HSC -ns 64 -re 256 -r1 3 -r2 3 -k -a 1 

- Cross : 
OUTDIR="outputs/sims2/cross_j64_ns256" && setcc && python run_corr.py -o $OUTDIR -s 5 -c 200 -t1 LRG -t2 HSC -ns 64 -re 256 -r1 3 -r2 3 && python run_corr.py -o $OUTDIR -s 5 -c 200 -t1 BGS_ANY -t2 HSC -ns 64 -re 256 -r1 3 -r2 3 && python run_corr.py -o $OUTDIR -s 5 -c 200 -t1 ELGnotqso -t2 HSC -ns 64 -re 256 -r1 3 -r2 3