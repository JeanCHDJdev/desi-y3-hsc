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
setcc && python run_corr.py -o results/autos_j64_ns256_NGC -s 0 -c 200 -t1 ELGnotqso -w PIP -k -ns 64 -re 256 -a 1 && python run_corr.py -o results/autos_j64_ns256_NGC -s 0 -c 200 -t1 LRG -w PIP -k  -ns 64 -re 256 -a 1
```

```bash
setcc && python run_corr.py -o results/autos_j64_ns256_NGC -s 0 -c 200 -t1 QSO -w PIP -k -ns 64 -re 256 -a 1 && python run_corr.py -o results/autos_j64_ns256_NGC -s 0 -c 200 -t1 BGS_ANY -w PIP -k -ns 64 -re 256 -a 1 -j
```

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
setcc && python run_corr.py -o outputs/results/autos_j64_ns256_zbin -s 0 -c 200 -t1 HSC -ns 64 -re 256 -r2 5 -z -j
```

##### HSC-DESI cross-correlation

--------------------------------

Same bins as previous settings. 
Command :
```bash
setcc && python run_corr.py -o outputs/results/cross_j64_ns256_zbin -s 0 -c 200 -t1 ELGnotqso -t2 HSC -ns 64 -re 256 -r2 15 -r2 2 -z -w PIP -j && python run_corr.py -o outputs/results/cross_j64_ns256_zbin -s 0 -c 200 -t1 LRG -t2 HSC -ns 64 -re 256 -r2 15 -r2 2 -z -w PIP -j
```