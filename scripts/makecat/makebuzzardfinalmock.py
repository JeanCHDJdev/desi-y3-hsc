#!/usr/bin/env python
# This code is located at: /project/projectdirs/desi/users/cblake/lensing
# First run: source /project/projectdirs/desi/software/desi_environment.sh

import sys
sys.path.insert(0,'/project/projectdirs/desi/mocks/desiqa/cori/lib/python3.6/site-packages/')
import numpy as np
import healpy as hp
from scipy.interpolate import splev,splrep
from scipy.spatial import cKDTree
from astropy.io import fits

def main(ireg):

# Parameters
  imock = 11          # Buzzard realisation to process
                      # Available mocks: 0,3,4,5,6,8,9,11
  isurvlst = [1,2,3]  # WL surveys to model (1=KiDS, 2=DES, 3=HSC)
  npix = 180          # Number of healpix pixels to generate (full is 180)
  dolens = True       # Generate lens catalogues
  dosource = True     # Generate source catalogues
  
# Initialisations
  nside = 8
  areamock = hp.nside2pixarea(nside,degrees=True)
  ipixlst = getpixlst()
  ipixlst = ipixlst[:npix]
  ipixlst = np.array([ipixlst[ireg-1]])
  zminsource,zmaxsource = 0.,2.

# Loop over pixels
  for ipix in ipixlst:
    print('\nProducing mock for pixel',str(ipix),'...')

# Generate lens catalogues
    if (dolens):
# Read in Buzzard BGS, LRG, ELG catalogues
      rasbgsdat,decbgsdat,zspecbgsdat,nbgsdat = readlensdat(imock,ipix,1)
      raslrgdat,declrgdat,zspeclrgdat,nlrgdat = readlensdat(imock,ipix,2)
#      raselgdat,decelgdat,zspecelgdat,nelgdat = readlensdat(imock,ipix,3)

# Make BGS, LRG, ELG random catalogues
      rasbgsran,decbgsran,zspecbgsran,nbgsran = makelensran(zspecbgsdat,ipix,areamock)
      raslrgran,declrgran,zspeclrgran,nlrgran = makelensran(zspeclrgdat,ipix,areamock)
#      raselgran,decelgran,zspecelgran,nelgran = makelensran(zspecelgdat,ipix,areamock)

# Generate source catalogues
    if (dosource):
# Read in Buzzard source catalogue
      rassource,decsource,zspecsource,gamma1source,gamma2source,gmagsource,rmagsource,imagsource,zmagsource,ymagsource = readsourcedat(imock,ipix)

# Cut source catalogue within redshift limits
      keep = (zspecsource > zminsource) & (zspecsource < zmaxsource)
      print('Original catalog has ',len(zspecsource),'sources')
      rassource,decsource,zspecsource,gamma1source,gamma2source,gmagsource,rmagsource,imagsource,zmagsource,ymagsource = rassource[keep],decsource[keep],zspecsource[keep],gamma1source[keep],gamma2source[keep],gmagsource[keep],rmagsource[keep],imagsource[keep],zmagsource[keep],ymagsource[keep]
      nsource = len(rassource)
      print('Cut to',nsource,'sources with',zminsource,'< z_spec <',zmaxsource)

# Loop over weak lensing surveys
      for isurv in isurvlst:

# Settings for KiDS data
        if (isurv == 1):
          print('\nGenerating tailored mock for KiDS...')
          gmagsurv,rmagsurv,imagsurv,zmagsurv,ymagsurv,weisurv,zphotsurv,nsurv,nsurv0 = readkidssurv()
          weical,zspeccal,zphotcal,ncal = readkidscal()
          areadata = 374.7
          ntom = 5
          zplims = np.array([0.1,0.3,0.5,0.7,0.9,1.2])
          magshift = np.array([0.,0.,0.,0.,0.])
          sigetom = np.array([0.274,0.271,0.289,0.287,0.301])
          mcorrtom = np.array([-0.009,-0.011,-0.015,0.002,0.007])

# Settings for DES Y3 data
        elif (isurv == 2):
          print('\nGenerating tailored mock for DES Y3...')
          gmagsurv,rmagsurv,imagsurv,zmagsurv,weisurv,r11surv,r22surv,zphotsurv,nsurv,nsurv0 = readdesy3surv()
          weical,zspeccal,zphotcal,ncal = readdesy3cal()
          areadata = 4143.
          ntom = 4
          zplims = np.array([0.,0.5,1.,1.5,2.])
          magshift = np.array([0.3,0.3,0.3,0.2])
          sigetom = np.array([0.201,0.204,0.195,0.203])
          mcorrtom = np.array([-0.006,-0.020,-0.024,-0.037])

# Settings for HSC data
        elif (isurv == 3):
          print('\nGenerating tailored mock for HSC...')
          gmagsurv,rmagsurv,imagsurv,zmagsurv,ymagsurv,weisurv,mcorrsurv,ermssurv,zphotsurv,nsurv,nsurv0 = readhscsurv()
          weical,zspeccal,zphotcal,ncal = readhsccal()
          areadata = 158.3
          ntom = 4
          zplims = np.array([0.3,0.6,0.9,1.2,1.5])
          magshift = np.array([0.,0.,0.,0.1])

# Construct spec-z vs phot-z scatter from calibration sample
        zsmin0,zsmax0,nzs,zpmin0,zpmax0,nzp = 0.,2.,40,0.,2.,40
        probzszp = getprobzszp(zspeccal,zphotcal,zsmin0,zsmax0,nzs,zpmin0,zpmax0,nzp)
# Apply this scatter pattern to mocks
        zphotsource = dophotzdraw(zspecsource,probzszp,zsmin0,zsmax0,nzs,zpmin0,zpmax0,nzp)
# Cut mock catalogue within photo-z limits
        zpmin,zpmax = zplims[0],zplims[-1]
        keep = (zphotsource > zpmin) & (zphotsource < zpmax)
        rassource1,decsource1,zspecsource1,zphotsource,gamma1source1,gamma2source1,gmagsource1,rmagsource1,imagsource1,zmagsource1,ymagsource1 = rassource[keep],decsource[keep],zspecsource[keep],zphotsource[keep],gamma1source[keep],gamma2source[keep],gmagsource[keep],rmagsource[keep],imagsource[keep],zmagsource[keep],ymagsource[keep]
        nsource = len(rassource1)
        print('Cut to',nsource,'sources with',zpmin,'< z_phot <',zpmax)
# Tomographic bins for catalogues
        itommock = np.digitize(zphotsource,zplims) - 1
        itomdata = np.digitize(zphotsurv,zplims) - 1

# Shift magnitudes to match number densities
        print('\nShifting magnitudes by',magshift)
        gmagsource1 -= magshift[itommock]
        rmagsource1 -= magshift[itommock]
        imagsource1 -= magshift[itommock]
        zmagsource1 -= magshift[itommock]
        ymagsource1 -= magshift[itommock]

# Sub-sample mock catalogue to match data magnitude distribution in each tomographic bin
        print('\nSub-sampling mock to match data magnitude distribution...')
        magmin,magmax,nmag = 18.,26.,80
        dmag = (magmax-magmin)/float(nmag)
        if (isurv == 3):
          ibinmagdata = np.digitize(imagsurv,np.linspace(magmin,magmax,nmag+1)) - 1
          ibinmagmock = np.digitize(imagsource1,np.linspace(magmin,magmax,nmag+1)) - 1
        else:
          ibinmagdata = np.digitize(rmagsurv,np.linspace(magmin,magmax,nmag+1)) - 1
          ibinmagmock = np.digitize(rmagsource1,np.linspace(magmin,magmax,nmag+1)) - 1
        ibinmagdata = np.where(ibinmagdata==-1,0,ibinmagdata)
        ibinmagdata = np.where(ibinmagdata==nmag,nmag-1,ibinmagdata)
        ibinmagmock = np.where(ibinmagmock==-1,0,ibinmagmock)
        ibinmagmock = np.where(ibinmagmock==nmag,nmag-1,ibinmagmock)
        keep = np.repeat(False,nsource)
        fdata = float(nsurv0)/float(nsurv)
        for itom in range(ntom):
          nsource1 = len(zphotsource[itommock==itom])
          magcomp = np.zeros(nmag)
          for imag in range(nmag):
            cutdata = (itomdata == itom) & (ibinmagdata == imag)
            cutmock = (itommock == itom) & (ibinmagmock == imag)
            ndensdata = fdata*float(len(zphotsurv[cutdata]))/areadata
            ndensmock = float(len(zphotsource[cutmock]))/areamock
            if (ndensmock > 0.):
              magcomp[imag] = ndensdata/ndensmock
          keep[itommock==itom] = (np.random.rand(nsource1) < magcomp[ibinmagmock[itommock==itom]])
        rassource1,decsource1,zspecsource1,zphotsource,gamma1source1,gamma2source1,gmagsource1,rmagsource1,imagsource1,zmagsource1,ymagsource1 = rassource1[keep],decsource1[keep],zspecsource1[keep],zphotsource[keep],gamma1source1[keep],gamma2source1[keep],gmagsource1[keep],rmagsource1[keep],imagsource1[keep],zmagsource1[keep],ymagsource1[keep]
        nsource = len(rassource1)
        print('Cut to',nsource,'sources')

# Find weight for Buzzard sources in nearest neighbour data using KDTree
        neigh = 3
        mcorrsource,ermssource,r11source,r22source = np.empty(nsource),np.empty(nsource),np.empty(nsource),np.empty(nsource)
        if (isurv == 1):
          magssource = np.dstack([gmagsource1,rmagsource1,imagsource1,zmagsource1,ymagsource1])[0]
          magssurv = np.dstack([gmagsurv,rmagsurv,imagsurv,zmagsurv,ymagsurv])[0]
          weisource,temp1,temp2 = findnearestsurv(magssource,magssurv,weisurv,weisurv,weisurv,neigh)
        elif (isurv == 2):
          magssource = np.dstack([gmagsource1,rmagsource1,imagsource1,zmagsource1])[0]
          magssurv = np.dstack([gmagsurv,rmagsurv,imagsurv,zmagsurv])[0]
          weisource,r11source,r22source = findnearestsurv(magssource,magssurv,weisurv,r11surv,r22surv,neigh)
        elif (isurv == 3):
          magssource = np.dstack([gmagsource1,rmagsource1,imagsource1,zmagsource1,ymagsource1])[0]
          magssurv = np.dstack([gmagsurv,rmagsurv,imagsurv,zmagsurv,ymagsurv])[0]
          weisource,mcorrsource,ermssource = findnearestsurv(magssource,magssurv,weisurv,mcorrsurv,ermssurv,neigh)

# Apply shape noise and calibration corrections
        if (isurv == 1):
# KiDS: x=mcorr, y=dummy
          e1source,e2source,g1source,g2source,xsource = appshapecalkids(gamma1source1,gamma2source1,zphotsource,mcorrtom,sigetom,zplims)
          ysource = np.zeros(nsource)
          csurv = '_kids'
        elif (isurv == 2):
# DES: x=R11, y=R22
          e1source,e2source,g1source,g2source = appshapecaldesy3(gamma1source1,gamma2source1,zphotsource,mcorrtom,r11source,r22source,sigetom,zplims)
          xsource,ysource = r11source,r22source
          csurv = '_desy3'
        elif (isurv == 3):
# HSC: x=mcorr, y=erms
          e1source,e2source,g1source,g2source = appshapecalhsc(gamma1source1,gamma2source1,weisource,mcorrsource,ermssource)
          xsource,ysource = mcorrsource,ermssource
          csurv = '_hsc'

# Write source catalogue in tomographic bins
        for itom in range(ntom):
          print('Generating source catalogue in tomographic bin',itom+1,'...')
          zmin,zmax = zplims[itom],zplims[itom+1]
          cut = (zphotsource > zmin) & (zphotsource < zmax)
          nsource1 = len(rassource1[cut])
          print('Cut to',nsource1,'sources in range',zmin,'< z <',zmax)
#          outfile = 'stage1mock_mock' + str(imock) + '_pix' + str(ipix) + '_sources' + csurv + '_tom' + str(itom+1) + '.fits'
          outfile = 'buzzard3x2_mock' + str(imock) + '_pix' + str(ipix) + '_sources' + csurv + '_tom' + str(itom+1) + '.fits'
          writesourcefits(outfile,rassource1[cut],decsource1[cut],zspecsource1[cut],zphotsource[cut],gamma1source1[cut],gamma2source1[cut],g1source[cut],g2source[cut],e1source[cut],e2source[cut],weisource[cut],xsource[cut],ysource[cut],isurv)

# Write lens data and random catalogues in tomographic bins
    if (dolens):
#      itommin,itommax = 0,8
      itommin,itommax = 0,4
      for itom in range(itommin,itommax):
        print('Generating lens catalogue in tomographic bin',itom+1)
        if (itom == 0):
          zmin,zmax = 0.1,0.3
          ext = '_BGS_zs0pt1_0pt3'
        elif (itom == 1):
          zmin,zmax = 0.3,0.5
          ext = '_BGS_zs0pt3_0pt5'
        elif (itom == 2):
          zmin,zmax = 0.5,0.7
          ext = '_LRG_zs0pt5_0pt7'
        elif (itom == 3):
          zmin,zmax = 0.7,0.9
          ext = '_LRG_zs0pt7_0pt9'
        elif (itom == 4):
          zmin,zmax = 0.9,1.1
          ext = '_ELG_zs0pt9_1pt1'
        elif (itom == 5):
          zmin,zmax = 1.1,1.3
          ext = '_ELG_zs1pt1_1pt3'
        elif (itom == 6):
          zmin,zmax = 1.3,1.5
          ext = '_ELG_zs1pt3_1pt5'
        elif (itom == 7):
          zmin,zmax = 1.5,1.7
          ext = '_ELG_zs1pt5_1pt7'
        if ((itom == 0) or (itom == 1)):
          raslensdat,declensdat,zspeclensdat = rasbgsdat,decbgsdat,zspecbgsdat
          raslensran,declensran,zspeclensran = rasbgsran,decbgsran,zspecbgsran
        elif ((itom == 2) or (itom == 3)):
          raslensdat,declensdat,zspeclensdat = raslrgdat,declrgdat,zspeclrgdat
          raslensran,declensran,zspeclensran = raslrgran,declrgran,zspeclrgran
        else:
          raslensdat,declensdat,zspeclensdat = raselgdat,decelgdat,zspecelgdat
          raslensran,declensran,zspeclensran = raselgran,decelgran,zspecelgran
        cut = (zspeclensdat > zmin) & (zspeclensdat < zmax)
        nlensdat1 = len(raslensdat[cut])
        weilensdat = np.ones(nlensdat1)
        print('Cut to',nlensdat1,'data lenses in range',zmin,'< z <',zmax)
#        outfile = 'stage1mock_mock' + str(imock) + '_pix' + str(ipix) + '_lenses' + ext + '.fits'
        outfile = 'buzzard3x2_mock' + str(imock) + '_pix' + str(ipix) + '_lenses' + ext + '.fits'
        writelensfits(outfile,raslensdat[cut],declensdat[cut],zspeclensdat[cut],weilensdat)
        cut = (zspeclensran > zmin) & (zspeclensran < zmax)
        nlensran1 = len(raslensran[cut])
        weilensran = np.ones(nlensran1)
        print('Cut to',nlensran1,'random lenses in range',zmin,'< z <',zmax)
#        outfile = 'stage1mock_mock' + str(imock) + '_pix' + str(ipix) + '_randlenses' + ext + '.fits'
        outfile = 'buzzard3x2_mock' + str(imock) + '_pix' + str(ipix) + '_randlenses' + ext + '.fits'
        writelensfits(outfile,raslensran[cut],declensran[cut],zspeclensran[cut],weilensran)
    
  return

########################################################################
# List of nside=8 healpix pixels in Buzzard quadrant.                  #
########################################################################

def getpixlst():
  ipixlst = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,279,283,284,285,286,287,305,308,309,310,311,317,343,347,348,349,350,351,359,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,423,427,428,429,430,431,434,440,441,442,443,446]
  return ipixlst

########################################################################
# Read in Buzzard lens data.                                           #
########################################################################

def readlensdat(imock,ipix,lensopt):
#  stem = '/project/projectdirs/desi/mocks/buzzard/buzzard_v2.0/buzzard-' + str(imock) + '/addgalspostprocess/desi_targets/'
# With corrected RSD
  stem = '/project/projectdirs/desi/mocks/buzzard/buzzard_v2.0/buzzard-' + str(imock) + '/addgalspostprocess/desi_targets_v1.0/'
  if (lensopt == 1):
    ctype = 'BGS'
    zmin,zmax = 0.1,0.5
  elif (lensopt == 2):
    ctype = 'LRG'
    zmin,zmax = 0.5,0.9
  elif (lensopt == 3):
    ctype = 'ELG'
    zmin,zmax = 0.9,1.7
  print('\nReading in DESI Buzzard',ctype,'data...')  
  buzzardfile = 'Chinchilla-' + str(imock)
  if (imock == 3):
    buzzardfile = buzzardfile + '_lensed_rs_shift_rs_scat_cam'
  elif (imock == 6):
    buzzardfile = buzzardfile + '_lensed_cam_rs_scat_shift'
  else:
    buzzardfile = buzzardfile + '_cam_rs_scat_shift_lensed'
  infile = stem + buzzardfile + '.' + str(ipix) + '.fits'
  print(infile)
  hdulist = fits.open(infile)
  table = hdulist[1].data
  ras = table.field('ra')
  dec = table.field('dec')
  red = table.field('z')
  if (lensopt == 1):
    keep = table.field('isbgs_bright')
  elif (lensopt == 2):
    keep = table.field('islrg')
  elif (lensopt == 3):
    keep = table.field('iselg')
  hdulist.close()
  keep = keep & (red > zmin) & (red < zmax)
  ras,dec,red = ras[keep],dec[keep],red[keep]
  ngal = len(ras)
  print('Cut to',ngal,'objects in',zmin,'< z <',zmax)
  return ras,dec,red,ngal

########################################################################
# Make random catalogue in a pixel.                                    #
########################################################################

def makelensran(dred,ipixgen,areamock):
  rsets = 30
  rmin,rmax,dmin,dmax = 0.,180.,0.,90.
  zmin,zmax,dzbin = np.amin(dred),np.amax(dred),0.01
  print('\nGenerating DESI Buzzard randoms for',zmin,'< z <',zmax)
  fact = np.pi/180.
  smin,smax = np.sin(fact*dmin),np.sin(fact*dmax)
  areagen = (rmax-rmin)*np.degrees(smax-smin)
  nran = int(float(rsets*len(dred))*(areagen/areamock))
  ras = rmin + (rmax-rmin)*np.random.rand(nran)
  dec = np.arcsin(smin + (smax-smin)*np.random.rand(nran))/fact
  theta,phi = np.radians(90.-dec),np.radians(ras)
  nside = 8
  ipix = hp.ang2pix(nside,theta,phi,nest=True)
  cut = (ipix == ipixgen)
  ras,dec = ras[cut],dec[cut]
  nran1 = len(ras)
  nz = int(np.rint((zmax-zmin)/dzbin))
  dz = (zmax-zmin)/nz
  nzmod,zmod = np.histogram(dred,bins=nz,range=[zmin,zmax],density=True)
  if (zmin < dz):
    zmod[1:] -= 0.5*dz
    nzmod = np.concatenate((np.array([0.]),nzmod))
  else:
    zmod = zmod[:-1] + 0.5*dz
  tck = splrep(zmod,nzmod)
  zs = np.arange(zmin,zmax,0.00001)
  Ns = np.cumsum(splev(zs,tck))
  Ns /= Ns[-1]
  N = np.random.rand(nran1)
  ind = np.searchsorted(Ns,N)
  red = zs[ind]  
  print(nran1,'total randoms')
  return ras,dec,red,nran1

########################################################################
# Read in Buzzard source data.                                         #
########################################################################

def readsourcedat(imock,ipix):
  print('\nReading in Buzzard source catalogue for realisation',imock,'pixel',ipix,'...')
  stem = '/project/projectdirs/desi/mocks/buzzard/buzzard_v2.0/buzzard-' + str(imock) + '/addgalspostprocess/'
  buzzardfile1 = 'truth/Chinchilla-' + str(imock)
  if (imock == 3):
    buzzardfile1 = buzzardfile1 + '_lensed_rs_shift_rs_scat_cam'
  elif (imock == 6):
    buzzardfile1 = buzzardfile1 + '_lensed_cam_rs_scat_shift'
  else:
    buzzardfile1 = buzzardfile1 + '_cam_rs_scat_shift_lensed'
  buzzardfile2 = 'surveymags/'
  if (imock == 0):
    buzzardfile2 = buzzardfile2 + 'surveymags-aux'
  else:
    buzzardfile2 = buzzardfile2 + 'Chinchilla-' + str(imock) + '-aux'
  infile = stem + buzzardfile1 + '.' + str(ipix) + '.fits'
  print(infile)
  hdulist = fits.open(infile)
  table = hdulist[1].data
  ras = table.field('RA')
  dec = table.field('DEC')
  gamma1 = table.field('GAMMA1')
  gamma2 = table.field('GAMMA2')
  hdulist.close()
  infile = stem + buzzardfile2 + '.' + str(ipix) + '.fits'
  print(infile)
  hdulist = fits.open(infile)
  table = hdulist[1].data
  zspec = table.field('Z')
  mags = table.field('LMAG')
  hdulist.close()
  gmag = mags[:,1]
  rmag = mags[:,2]
  imag = mags[:,3]
  zmag = mags[:,4]
  ymag = mags[:,5]
  return ras,dec,zspec,gamma1,gamma2,gmag,rmag,imag,zmag,ymag

########################################################################
# Read in KiDS source magnitudes and weights.                          #
########################################################################

def readkidssurv():
  nsamp = 1000000
  stem = 'lenscats/'
  infile = stem + 'kids_mag.fits'
  print('\nReading in KiDS source catalogue...')
  print(infile)
  hdulist = fits.open(infile)
  table = hdulist[1].data
  gmag = table.field('MAG_GAAP_g')
  rmag = table.field('MAG_GAAP_r')
  imag = table.field('MAG_GAAP_i')
  zmag = table.field('MAG_GAAP_Z')
  ymag = table.field('MAG_GAAP_Y')
  wei = table.field('weight')
  zb = table.field('Z_B')
  hdulist.close()
  cut = (zb > 0.1) & (zb < 1.2)
  gmag,rmag,imag,zmag,ymag,wei,zb = gmag[cut],rmag[cut],imag[cut],zmag[cut],ymag[cut],wei[cut],zb[cut]
  ngal0 = len(gmag)
  print(ngal0,'KiDS sources read in')
  cut = np.random.choice(ngal0,nsamp,replace=False)
  gmag,rmag,imag,zmag,ymag,wei,zb = gmag[cut],rmag[cut],imag[cut],zmag[cut],ymag[cut],wei[cut],zb[cut]
  ngal = len(gmag)
  print('Cut to',ngal,'sources')
  zb += np.random.uniform(-0.005,0.005,ngal)
  return gmag,rmag,imag,zmag,ymag,wei,zb,ngal,ngal0

########################################################################
# Read in KiDS calibration data.                                       #
########################################################################

def readkidscal():
  stem = 'lenscats/'
  infile = stem + 'kids_cal.fits'
  print('\nReading in KiDS calibration sample...')
  print(infile)
  hdulist = fits.open(infile)
  table = hdulist[1].data
  zspec = table.field('z_spec')
  zb = table.field('z_B')
  wei = table.field('spec_weight_CV') # includes lensfit weight
  hdulist.close()
  ngal = len(zspec)
  print(ngal,'calibration sources read in')
  zb += np.random.uniform(-0.005,0.005,ngal)
  return wei,zspec,zb,ngal

########################################################################
# Read in DES Y3 metacal source magnitudes and weights.                #
########################################################################

def readdesy3surv():
  nsamp = 1000000
  stem = 'lenscats/'
  infile = stem + 'desy3_mag.fits'
  print('\nReading in DES Y3 source catalogue...')
  print(infile)
  hdulist = fits.open(infile)
  table = hdulist[1].data
  gmag = table.field('mag_g')
  rmag = table.field('mag_r')
  imag = table.field('mag_i')
  zmag = table.field('mag_z')
  wei = table.field('wei')
  r11 = table.field('R11')
  r22 = table.field('R22')
  itom = table.field('tombin').astype('float')
  hdulist.close()
  ngal0 = len(rmag)
  print(ngal0,'DES Y3 sources read in')
  zphot = np.random.uniform(0.5*itom,0.5*(itom+1))
  cut = np.random.choice(ngal0,nsamp,replace=False)
  gmag,rmag,imag,zmag,wei,r11,r22,zphot = gmag[cut],rmag[cut],imag[cut],zmag[cut],wei[cut],r11[cut],r22[cut],zphot[cut]
  ngal = len(rmag)
  print('Cut to',ngal,'sources')
  return gmag,rmag,imag,zmag,wei,r11,r22,zphot,ngal,ngal0

########################################################################
# Read in DES Y3 metacal calibration data.                             #
########################################################################

def readdesy3cal():
  stem = 'lenscats/'
  infile = stem + 'desy3_cal.fits'
  print('\nReading in DES Y3 calibration sample...')
  print(infile)
  hdulist = fits.open(infile)
  table = hdulist[1].data
  zphot = table.field('zphot')
  zspec = table.field('zspec')
  hdulist.close()
  ngal = len(zphot)
  wei = np.ones(ngal)
  print(ngal,'calibration sources read in')
  return wei,zspec,zphot,ngal

########################################################################
# Read in HSC source magnitudes and weights.                           #
########################################################################

def readhscsurv():
  nsamp = 1000000
  stem = 'lenscats/'
  infile = stem + 'hsc_mag.fits'
  print('\nReading in HSC source catalogue...')
  print(infile)
  hdulist = fits.open(infile)
  table = hdulist[1].data
  gmag = table.field('gcmodel_mag')
  rmag = table.field('rcmodel_mag')
  imag = table.field('icmodel_mag')
  zmag = table.field('zcmodel_mag')
  ymag = table.field('ycmodel_mag')
  wei = table.field('weight')
  mcorr = table.field('mcorr')
  erms = table.field('erms')
  zphot = table.field('photoz_best')
  hdulist.close()
  ngal = len(gmag)
  cut = (zphot > 0.3) & (zphot < 1.5)
  gmag,rmag,imag,zmag,ymag,wei,mcorr,erms,zphot = gmag[cut],rmag[cut],imag[cut],zmag[cut],ymag[cut],wei[cut],mcorr[cut],erms[cut],zphot[cut]
  ngal0 = len(gmag)
  print(ngal0,'HSC sources read in')
  cut = np.random.choice(ngal0,nsamp,replace=False)
  gmag,rmag,imag,zmag,ymag,wei,mcorr,erms,zphot = gmag[cut],rmag[cut],imag[cut],zmag[cut],ymag[cut],wei[cut],mcorr[cut],erms[cut],zphot[cut]
  ngal = len(gmag)
  print('Cut to',ngal,'sources')
  return gmag,rmag,imag,zmag,ymag,wei,mcorr,erms,zphot,ngal,ngal0

########################################################################
# Read in HSC calibration data.                                        #
########################################################################

def readhsccal():
  stem = 'lenscats/'
  infile = stem + 'hsc_cal.fits'
  print('\nReading in HSC calibration sample...')
  print(infile)
  hdulist = fits.open(infile)
  table = hdulist[1].data
  zphot = table.field('redhsc')
  zspec = table.field('redcosmos')
  weisom = table.field('weisom')
  weilens = table.field('weilens')
  hdulist.close()
  ngal = len(zphot)
  print(ngal,'calibration sources read in')
  wei = weisom*weilens
  return wei,zspec,zphot,ngal

########################################################################
# Get 2D (z_phot,z_spec) distribution.                                 #
########################################################################

def getprobzszp(zspec,zphot,zsmin,zsmax,nzs,zpmin,zpmax,nzp):
  print('\nGet (z_phot,z_spec) probability distribution...')
  probzszp,edges = np.histogramdd(np.vstack([zspec,zphot]).transpose(),bins=(nzs,nzp),range=((zsmin,zsmax),(zpmin,zpmax)))
  return probzszp

########################################################################
# Draw photo-z values from 2D (z_phot,z_spec) distribution.            #
########################################################################

def dophotzdraw(zspec,probzszp,zsmin,zsmax,nzs,zpmin,zpmax,nzp):
  print('\nDrawing photo-z values from probability distribution...')
  dzp = (zpmax-zpmin)/float(nzp)
  zpcen = np.linspace(zpmin+0.5*dzp,zpmax-0.5*dzp,nzp)
  zpmod = zpcen
  zphotarr = np.arange(zpmin,zpmax,0.0001)
  zsbin = np.digitize(zspec,np.linspace(zsmin,zsmax,nzs+1)) - 1
  zphot = np.zeros(len(zspec))
  for izs in range(nzs):
    zsind = (zsbin == izs)
    ngal = len(zspec[zsind])
    pzmod = probzszp[izs,:]
    tck = splrep(zpmod,pzmod)
    Ns = np.cumsum(splev(zphotarr,tck))
    Ns /= Ns[-1]
    N = np.random.rand(ngal)
    zpind = np.searchsorted(Ns,N)
    zphot[zsind] = zphotarr[zpind]
  return zphot

########################################################################
# Assign source properties by randomly drawing amongst the "neigh"     #
# nearest neighbours using a KDTree in magnitude space.                #
########################################################################

def findnearestsurv(mags,magssrc,p1src,p2src,p3src,neigh):
  print('\nAssigning properties from',neigh,'nearest neighbours...')
# Setting up KDTree on KiDS magnitudes
  tree = cKDTree(magssrc)
# Indices of nearest neighbours for each Buzzard magnitude
  ineighlst = tree.query(mags,k=neigh)[1]
# Random draw from neigh indices
  ineighdraw = np.random.randint(0,neigh,size=len(ineighlst))
  nmock = mags.shape[0]
# Assign properties
  p1,p2,p3 = np.empty(nmock),np.empty(nmock),np.empty(nmock)
  for i,j in enumerate(ineighdraw):
    k = ineighlst[i,j]
    p1[i],p2[i],p3[i] = p1src[k],p2src[k],p3src[k]
  return p1,p2,p3

########################################################################
# Apply shape noise to simulated catalogue including shear bias.       #
########################################################################

def appshapecalkids(gamma1,gamma2,zphot,mcorrtom,sigetom,zplims):
  print('\nApplying KiDS shape noise and calibration bias...')
  ngal = len(gamma1)
  ibin = np.digitize(zphot,zplims) - 1
  mcorr = mcorrtom[ibin]
  sige = sigetom[ibin]
  g1 = gamma1*(1.+mcorr)
  g2 = gamma2*(1.+mcorr)
  n1 = sige*np.random.normal(size=ngal)
  n2 = sige*np.random.normal(size=ngal)
  a1 = g1 + n1
  a2 = g2 + n2
  a3 = 1. + g1*n1 + g2*n2
  a4 = g1*n2 - g2*n1
  e1 = (a1*a3 + a2*a4)/(a3*a3 + a4*a4)
  e2 = (a2*a3 - a1*a4)/(a3*a3 + a4*a4)
  return e1,e2,g1,g2,mcorr

def appshapecaldesy3(gamma1,gamma2,zphot,mcorrtom,r11,r22,sigetom,zplims):
  print('\nApplying DES Y3 shape noise and calibration bias...')
  ngal = len(gamma1)
  ibin = np.digitize(zphot,zplims) - 1
  mcorr = mcorrtom[ibin]
  sige = sigetom[ibin]
  g1 = gamma1*0.5*(r11+r22)*(1.+mcorr)
  g2 = gamma2*0.5*(r11+r22)*(1.+mcorr)
  n1 = sige*np.random.normal(size=ngal)
  n2 = sige*np.random.normal(size=ngal)
  a1 = g1 + n1
  a2 = g2 + n2
  a3 = 1. + g1*n1 + g2*n2
  a4 = g1*n2 - g2*n1
  e1 = (a1*a3 + a2*a4)/(a3*a3 + a4*a4)
  e2 = (a2*a3 - a1*a4)/(a3*a3 + a4*a4)
  return e1,e2,g1,g2

def appshapecalhsc(gamma1,gamma2,wei,mcorr,erms):
  print('\nApplying HSC shape noise and calibration bias...')
  ngal = len(gamma1)
  r = 1. - erms**2
  sige = 1./np.sqrt(wei)
  g1 = gamma1*2.*r*(1.+mcorr)
  g2 = gamma2*2.*r*(1.+mcorr)
  n1 = sige*np.random.normal(size=ngal)
  n2 = sige*np.random.normal(size=ngal)
  a1 = g1 + n1
  a2 = g2 + n2
  a3 = 1. + g1*n1 + g2*n2
  a4 = g1*n2 - g2*n1
  e1 = (a1*a3 + a2*a4)/(a3*a3 + a4*a4)
  e2 = (a2*a3 - a1*a4)/(a3*a3 + a4*a4)
  return e1,e2,g1,g2

########################################################################
# Write out source file as fits catalogue.                             #
########################################################################

def writesourcefits(outfile,ras,dec,zspec,zphot,gamma1,gamma2,g1,g2,e1,e2,wei,x,y,isurv):
  if (isurv == 1):
    print('\nWriting out KiDS source mock fits catalogue...')
  elif (isurv == 2):
    print('\nWriting out DES Y3 source mock fits catalogue...')
  elif (isurv == 3):
    print('\nWriting out HSC source mock fits catalogue...')
  print(outfile)
  col1 = fits.Column(name='RA',format='E',array=ras)
  col2 = fits.Column(name='Dec',format='E',array=dec)
  col3 = fits.Column(name='z_spec',format='E',array=zspec)
  col4 = fits.Column(name='z_phot',format='E',array=zphot)
  col5 = fits.Column(name='gamma_1',format='E',array=gamma1)
  col6 = fits.Column(name='gamma_2',format='E',array=gamma2)
  col7 = fits.Column(name='g_1',format='E',array=g1)
  col8 = fits.Column(name='g_2',format='E',array=g2)
  col9 = fits.Column(name='e_1',format='E',array=e1)
  col10 = fits.Column(name='e_2',format='E',array=e2)
  col11 = fits.Column(name='wei',format='E',array=wei)
  if (isurv == 1):
    col12 = fits.Column(name='m',format='E',array=x)
    col13 = fits.Column(name='dummy',format='E',array=y)
  elif (isurv == 2):
    col12 = fits.Column(name='R11',format='E',array=x)
    col13 = fits.Column(name='R22',format='E',array=y)
  elif (isurv == 3):
    col12 = fits.Column(name='m',format='E',array=x)
    col13 = fits.Column(name='erms',format='E',array=y)
  cols = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13])
  hdulist = fits.BinTableHDU.from_columns(cols)
  hdulist.writeto(outfile)
  return

########################################################################
# Write out lens file as fits catalogue.                               #
########################################################################

def writelensfits(outfile,ras,dec,zspec,wei):
  print('\nWriting out lens mock fits catalogue...')
  print(outfile)
  col1 = fits.Column(name='RA',format='E',array=ras)
  col2 = fits.Column(name='Dec',format='E',array=dec)
  col3 = fits.Column(name='z_spec',format='E',array=zspec)
  col4 = fits.Column(name='wei',format='E',array=wei)
  cols = fits.ColDefs([col1,col2,col3,col4])
  hdulist = fits.BinTableHDU.from_columns(cols)
  hdulist.writeto(outfile)
  return

if __name__ == '__main__':
  main(int(sys.argv[1]))
