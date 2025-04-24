#!/usr/bin/env python
#
# To run on NERSC:
# This code is located at: /project/projectdirs/desi/users/cblake/lensing
# First run: source /project/projectdirs/desi/software/desi_environment.sh
#
# To run on OzStar:
# This code is located at: /fred/oz073/cblake/buzzardcorr

import sys
# For NERSC
sys.path.insert(0, '/project/projectdirs/desi/mocks/desiqa/cori/lib/python3.6/site-packages/')
import numpy as np
from astropy.io import fits
import treecorr

def main(ireg):

########################################################################
# Parameters.                                                          #
########################################################################

  datopt = 2     # 1) 180 pixels 2) 89 regions
  istage = 1     # Stage of mock challenge
  statopt = 2    # 1) shear-shear 2) shear-galaxy 3) galaxy-galaxy
  imock = 11     # Buzzard realisation to process
  rsets = 30     # Number of random sets
  binslopgg,binslopng,binslopnn = 0.08,0.05,0.03
#  thmin,thmax,nthbin = 0.003,3.,15
#  thmin,thmax,nthbin = 0.01,10.,15

  if (datopt == 1):
    ipixlst = getpixlst()
    ireg1 = ipixlst[ireg-1]
  elif (datopt == 2):
    isurvlst = np.concatenate((np.repeat(0,20),np.repeat(1,9),np.repeat(2,60)))
    ireglst = np.concatenate((np.arange(20),np.arange(9),np.arange(60))) + 1
    isurv1 = isurvlst[ireg-1]
    ireg1 = ireglst[ireg-1]

# Stage 0 mocks
  if (istage == 0):
    stem = '/project/projectdirs/desi/users/cblake/lensing/stage0mocks_catalogues/'
    cstage = 'stage0mock_mock' + str(imock) + '_reg' + str(ireg1)
    nsurv = 1
    csurv = ['']
    ntomsurv = [4]
    exttomsurv = [['_zp0pt5_0pt7','_zp0pt7_0pt9','_zp0pt9_1pt1','_zp1pt1_1pt5']]
    nlens = 4
    extlens = ['_BGS_zs0pt1_0pt3','_BGS_zs0pt3_0pt5','_LRG_zs0pt5_0pt7','_LRG_zs0pt7_0pt9']
    thminsurv = [0.003]
    thmaxsurv = [3.]
    nthbinsurv = [15]
# Stage 1 mocks
  elif (istage == 1):
# For NERSC
#    stem = '/project/projectdirs/desi/users/cblake/lensing/stage1mocks_catalogues/'
    stem = '/project/projectdirs/desi/users/cblake/lensing/buzzard3x2_catalogues/'
# For OzStar
#    stem = '/fred/oz073/cblake/buzzard_mocks/'
    if (datopt == 1):
#      cstage = 'stage1mock_mock' + str(imock) + '_pix' + str(ireg1)
      cstage = 'buzzard3x2_mock' + str(imock) + '_pix' + str(ireg1)
      nsurv = 3
    elif (datopt == 2):
#      cstage = 'stage1mock_mock' + str(imock) + '_reg' + str(ireg1)
      cstage = 'buzzard3x2_mock' + str(imock) + '_reg' + str(ireg1)
      nsurv = 1
    csurv = ['_kids','_desy3','_hsc']
    ntomsurv = [5,4,4]
#    exttomsurv = [['_zp0pt10_0pt30','_zp0pt30_0pt50','_zp0pt50_0pt70','_zp0pt70_0pt90','_zp0pt90_1pt20'],['_zp0pt20_0pt43','_zp0pt43_0pt63','_zp0pt63_0pt90','_zp0pt90_1pt30',''],['_zp0pt30_0pt60','_zp0pt60_0pt90','_zp0pt90_1pt20','_zp1pt20_1pt50','']]
    exttomsurv = [['_tom1','_tom2','_tom3','_tom4','_tom5'],['_tom1','_tom2','_tom3','_tom4'],['_tom1','_tom2','_tom3','_tom4']]
    nlens = 4
    extlens = ['_BGS_zs0pt1_0pt3','_BGS_zs0pt3_0pt5','_LRG_zs0pt5_0pt7','_LRG_zs0pt7_0pt9']
    thminsurv = [0.5/60.,2.5/60.,(10.**0.05)/60.]
    thmaxsurv = [300./60.,250./60.,(10.**2.25)/60.]
    nthbinsurv = [9,20,22]

########################################################################
# Calculate tomographic shear correlation functions.                   #
########################################################################

  if (statopt == 1):
    ext = '_treecorr_bs0pt08.dat'
#    ext = '_nonoise_treecorr_bs0pt08.dat'
#    ext = '_distnonoise_treecorr_bs0pt08.dat'
    print('\nRunning treecorr for cosmic shear...')
    for isurv in range(nsurv):
      if (datopt == 1):
        isurv1 = isurv
      thmin,thmax,nthbin = thminsurv[isurv1],thmaxsurv[isurv1],nthbinsurv[isurv1]
      print('Using thmin =',60.*thmin,'thmax =',60.*thmax,'nthbin =',nthbin)
      for itom in range(ntomsurv[isurv1]):
        sourcefile = cstage + '_sources' + csurv[isurv1] + exttomsurv[isurv1][itom] + '.fits'
        rassource,decsource,e1,e2,weisource,nsource = readsourcefits(stem+sourcefile)
        cat1 = treecorr.Catalog(ra=rassource,dec=decsource,g1=e1,g2=e2,w=weisource,ra_units='deg',dec_units='deg',flip_g2=True)
        for jtom in range(itom,ntomsurv[isurv1]):
          gg = treecorr.GGCorrelation(min_sep=thmin,max_sep=thmax,nbins=nthbin,sep_units='deg',bin_slop=binslopgg)
          if (jtom == itom):
            print('Computing shear auto-correlation function for itom',itom+1,'...')
            gg.process(cat1)
          else:
            sourcefile = cstage + '_sources' + csurv[isurv1] + exttomsurv[isurv1][jtom] + '.fits'
            rassource,decsource,e1,e2,weisource,nsource = readsourcefits(stem+sourcefile)
            cat2 = treecorr.Catalog(ra=rassource,dec=decsource,g1=e1,g2=e2,w=weisource,ra_units='deg',dec_units='deg',flip_g2=True)
            print('Computing shear cross-correlation function for itom =',itom+1,'and jtom =',jtom+1,'...')
            gg.process(cat1,cat2)
          outfile = 'xipm_' + cstage + csurv[isurv1] + '_tom' + str(itom+1) + 'tom' + str(jtom+1) + ext
          print('Outputting data...')
          print(outfile)
          gg.write(outfile)

########################################################################
# Calculate galaxy-galaxy lensing correlation functions.               #
########################################################################

  elif (statopt == 2):
    ext = '_treecorr_bs0pt05.dat'
#    ext = '_nonoise_treecorr_bs0pt05.dat'
#    ext = '_distnonoise_treecorr_bs0pt05.dat'
    print('\nRunning treecorr for GGL...')
    dodat = True
    doran = True
    doboost = True
    dowei = True
    for isurv in range(nsurv):
      if (datopt == 1):
        isurv1 = isurv
      thmin,thmax,nthbin = thminsurv[isurv1],thmaxsurv[isurv1],nthbinsurv[isurv1]
      print('Using thmin =',60.*thmin,'thmax =',60.*thmax,'nthbin =',nthbin)
      for itom in range(ntomsurv[isurv1]):
        sourcefile = cstage + '_sources' + csurv[isurv1] + exttomsurv[isurv1][itom] + '.fits'
        rassource,decsource,e1,e2,weisource,nsource = readsourcefits(stem+sourcefile)
        cat2 = treecorr.Catalog(ra=rassource,dec=decsource,g1=e1,g2=e2,w=weisource,ra_units='deg',dec_units='deg',flip_g2=True)
        for ilens in range(nlens):
          if (dodat or doboost or dowei):
            if (datopt == 1):
              lensdatfile = cstage + '_lenses' + extlens[ilens] + '.fits'
            elif (datopt == 2):
              lensdatfile = cstage + '_lenses' + csurv[isurv1] + extlens[ilens] + '.fits'
            raslensdat,declensdat,weilensdat,nlensdat = readlensfits(stem+lensdatfile)
            cat1 = treecorr.Catalog(ra=raslensdat,dec=declensdat,w=weilensdat,ra_units='deg',dec_units='deg')
            if (dodat):
              print('Computing galaxy-shear cross-correlation function for itom =',itom+1,'and ilens =',ilens+1)
              ng = treecorr.NGCorrelation(min_sep=thmin,max_sep=thmax,nbins=nthbin,sep_units='deg',bin_slop=binslopng)
              ng.process(cat1,cat2)
              outfile = 'gt_' + cstage + csurv[isurv1] + '_lens' + str(ilens+1) + 'tom' + str(itom+1) + ext
              print('Outputting data...')
              print(outfile)
              ng.write(outfile)
            if (doboost):
              print('Computing galaxy-source cross-correlation function for itom =',itom+1,'and ilens =',ilens+1)
              nn = treecorr.NNCorrelation(min_sep=thmin,max_sep=thmax,nbins=nthbin,sep_units='deg',bin_slop=binslopng)
              nn.process(cat1,cat2)
              outfile = 'nlns_' + cstage + csurv[isurv1] + '_lens' + str(ilens+1) + 'tom' + str(itom+1) + ext
              print('Outputting data...')
              print(outfile)
              nn.write(outfile)
          if (doran or doboost or dowei):
            if (datopt == 1):
              lensranfile = cstage + '_randlenses' + extlens[ilens] + '.fits'
            elif (datopt == 2):
              lensranfile = cstage + '_randlenses' + csurv[isurv1] + extlens[ilens] + '.fits'
            raslensran,declensran,weilensran,nlensran = readlensfits(stem+lensranfile)
            if (rsets < 30):
              nsub = rsets*nlensdat
              raslensran,declensran,weilensran,nlensran = dosubdata(raslensran,declensran,weilensran,nsub)
            cat1 = treecorr.Catalog(ra=raslensran,dec=declensran,w=weilensran,ra_units='deg',dec_units='deg')
            if (doran):
              print('Computing random-shear cross-correlation function for itom =',itom+1,'and ilens =',ilens+1)
              ng = treecorr.NGCorrelation(min_sep=thmin,max_sep=thmax,nbins=nthbin,sep_units='deg',bin_slop=binslopng)
              ng.process(cat1,cat2)
              outfile = 'gtrand_' + cstage + csurv[isurv1] + '_lens' + str(ilens+1) + 'tom' + str(itom+1) + ext
              print('Outputting data...')
              print(outfile)
              ng.write(outfile)
            if (doboost):
              print('Computing random-source cross-correlation function for itom =',itom+1,'and ilens =',ilens+1)
              nn = treecorr.NNCorrelation(min_sep=thmin,max_sep=thmax,nbins=nthbin,sep_units='deg',bin_slop=binslopng)
              nn.process(cat1,cat2)
              outfile = 'nrns_' + cstage + csurv[isurv1] + '_lens' + str(ilens+1) + 'tom' + str(itom+1) + ext
              print('Outputting data...')
              print(outfile)
              nn.write(outfile)
            if (dowei):
              outfile = 'sumwei_' + cstage + csurv[isurv1] + '_lens' + str(ilens+1) + 'tom' + str(itom+1) + ext
              print('Outputting data...')
              print(outfile)
              f = open(outfile,'w')
              f.write('# sum w_lens,D, sum w_lens,R\n')
              f.write('{} {}'.format(np.sum(weilensdat),np.sum(weilensran)) + '\n')
              f.close()

########################################################################
# Calculate galaxy correlation functions.                              #
########################################################################

  elif (statopt == 3):
    ext = '_treecorr_bs0pt03.dat'
#    ext = '_treecorr_bs0pt03_bin2.dat'
    print('\nRunning treecorr for galaxy clustering...')
    for isurv in range(nsurv):
      if (datopt == 1):
        isurv1 = isurv
      thmin,thmax,nthbin = thminsurv[isurv1],thmaxsurv[isurv1],nthbinsurv[isurv1]
      print('Using thmin =',60.*thmin,'thmax =',60.*thmax,'nthbin =',nthbin)
#      for ilens in range(nlens):
      for ilens in range(3,nlens):
        if (datopt == 1):
          lensdatfile = cstage + '_lenses' + extlens[ilens] + '.fits'
          lensranfile = cstage + '_randlenses' + extlens[ilens] + '.fits'
        elif (datopt == 2):
          lensdatfile = cstage + '_lenses' + csurv[isurv1] + extlens[ilens] + '.fits'
          lensranfile = cstage + '_randlenses' + csurv[isurv1] + extlens[ilens] + '.fits'
        raslensdat,declensdat,weilensdat,nlensdat = readlensfits(stem+lensdatfile)
        raslensran,declensran,weilensran,nlensran = readlensfits(stem+lensranfile)
        if (rsets < 30):
          nsub = rsets*nlensdat
          raslensran,declensran,weilensran,nlensran = dosubdata(raslensran,declensran,weilensran,nsub)
        cat1 = treecorr.Catalog(ra=raslensdat,dec=declensdat,w=weilensdat,ra_units='deg',dec_units='deg')
        cat2 = treecorr.Catalog(ra=raslensran,dec=declensran,w=weilensran,ra_units='deg',dec_units='deg')
        nn = treecorr.NNCorrelation(min_sep=thmin,max_sep=thmax,nbins=nthbin,sep_units='deg',bin_slop=binslopnn)
        dr = treecorr.NNCorrelation(min_sep=thmin,max_sep=thmax,nbins=nthbin,sep_units='deg',bin_slop=binslopnn)
        rr = treecorr.NNCorrelation(min_sep=thmin,max_sep=thmax,nbins=nthbin,sep_units='deg',bin_slop=binslopnn)
        print('Computing DD for ilens =',ilens+1)
        nn.process(cat1)
        print('Computing DR for ilens =',ilens+1)
        dr.process(cat1,cat2)
        print('Computing RR for ilens =',ilens+1)
        rr.process(cat2)
        if (datopt == 1):
          outfile = 'wth_' + cstage + '_lens' + str(ilens+1) + ext
        elif (datopt == 2):
          outfile = 'wth_' + cstage + csurv[isurv1] + '_lens' + str(ilens+1) + ext
        print('Outputting data...')
        print(outfile)
        nn.write(outfile,rr,dr)

  return

def readsourcefits(infile):
  print('\nReading in source simulation...')
  print(infile)
  hdulist = fits.open(infile)
  table = hdulist[1].data
  ras = table.field('RA')
  dec = table.field('Dec')
#  e1 = table.field('gamma_1')
#  e2 = table.field('gamma_2')
#  e1 = table.field('g_1')
#  e2 = table.field('g_2')
  e1 = table.field('e_1')
  e2 = table.field('e_2')
  wei = table.field('wei')
  hdulist.close()
  ngal = len(ras)
  print(ngal,'sources read')
  print(e1)
  print(e2)
  return ras,dec,e1,e2,wei,ngal

def readlensfits(infile):
  print('\nReading in lens data...')
  print(infile)
  hdulist = fits.open(infile)
  table = hdulist[1].data
  ras = table.field('RA')
  dec = table.field('Dec')
  wei = table.field('wei')
  hdulist.close()
  ngal = len(ras)
  print(ngal,'lenses read in')
  return ras,dec,wei,ngal

def dosubdata(ras,dec,wei,nsub):
  ind = np.random.choice(len(ras),nsub,replace=False)
  ras,dec,wei = ras[ind],dec[ind],wei[ind]
  print('Sub-sampled to',nsub,'objects')
  return ras,dec,wei,nsub

def getpixlst():
  ipixlst = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,279,283,284,285,286,287,305,308,309,310,311,317,343,347,348,349,350,351,359,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,423,427,428,429,430,431,434,440,441,442,443,446]
  return ipixlst

if __name__ == '__main__':
  main(int(sys.argv[1]))
