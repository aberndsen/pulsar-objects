"""
A module providing access to the ATNF pulsar list, and creates 
aipy RadioFixedBodies but accomodates signal pulsations.

"""
import aipy as a
import urllib, os, datetime, numpy as np, ephem
import scipy, scipy.integrate
import subprocess,tempfile 

from numpy.random import normal,randint


###### Routines to generate/return lists of pulsars #######

def allpulsars():
        """
        Return a list of all pulsars in the ATNF catalogue.
        Get data from the local file 'pulsar_list.txt' provided
        it isn't older than 20 days; otherwise try contacting 
        the ATNF website.
        """
        
        tnow = datetime.datetime.today()
        try:
            tdata = datetime.datetime.fromtimestamp(
                    os.path.getmtime('pulsar_list.txt.bz2')
                    )
        except:
            tdata = datetime.datetime(1977,1,1)
        dt = tnow - tdata
        if dt.days > 20:
            try:
                srcs = ATNFpulsarlist(save=True) #update pulsar_list.txt
            except:
                data = np.genfromtxt('pulsar_list.txt',dtype='str')
                srcs = __data2pulsarlist__(data=data)                
        else:
            print "pulsars.py: using local pulsar list\n"
            data = np.genfromtxt('pulsar_list.txt.bz2',dtype='str')
            srcs = __data2pulsarlist__(data=data)

        allpulsars = PSrcCatalog(srcs)#a.amp.SrcCatalog(srcs)

        return allpulsars
        
                                  
def apulsar(src=None,dct=None):
    """
    Return the specified source,
    otherwise return a random pulsar.
    
    src = 'name', if this fails return by psrj name

    """
    plist = allpulsars()
    
    if src is None: 
        n = randint(0,len(plist))
        return [pobj for pobj in plist.itervalues() if pobj.props['indx'] == n]
    else:
        if plist.has_key(src):
            return [plist[src]]
	else:
            toret = [pobj for pobj in plist.itervalues() if pobj.props['psrj'] == src]
	    if len(toret) > 0:
                return [pobj for pobj in plist.itervalues() if pobj.props['psrj'] == src]     
        
    if dct is not None:
        for k,v in dct.iteritems():
            toret = [pobj for pobj in plist.itervalues() if pobj.props[k] == v]
	return toret
    
def ATNFpulsarlist(save=False):
        """
        Contact the ATNF pulsar catalogue, returning an array of data.
        Each row corresponding to one pulsar, with columns in the format:

        """

        try:
#URL to get |indx|Name|PSRJ|RA|DEC|posepoch|F0|F1|F2|pepoch|DM|W50|W10|S400|S1400|SPINDX
            url = 'http://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php?Name=Name&JName=JName&RaJ=RaJ&DecJ=DecJ&PosEpoch=PosEpoch&F0=F0&F1=F1&F2=F2&PEpoch=PEpoch&DM=DM&W50=W50&W10=W10&S400=S400&S1400=S1400&SPINDX=SPINDX&startUserDefined=true&c1_val=&c2_val=&c3_val=&c4_val=&sort_attr=jname&sort_order=asc&condition=&pulsar_names=&ephemeris=short&coords_unit=raj/decj&radius=&coords_1=&coords_2=&style=Short+without+errors&no_value=*&fsize=3&x_axis=&x_scale=linear&y_axis=&y_scale=linear&state=query&table_bottom.x=43&table_bottom.y=24'
            Hurl = '#|indx|Name|PSRJ|RA|DEC|posepoch|F0|F1|F2|pepoch|DM|W50|W10|S400|S1400|SPINDX\n'

#URL to get |NAME|PSRJ|RAJ|DECJ|POSEPOCH|F0|F1|F2|PEPOCH|DM|W50|W10|S400|S1400|SPINDX|PSRTYPE|NGLT|
            url2 = 'http://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php?Name=Name&JName=JName&RaJ=RaJ&DecJ=DecJ&PosEpoch=PosEpoch&F0=F0&F1=F1&F2=F2&PEpoch=PEpoch&DM=DM&W50=W50&W10=W10&S400=S400&S1400=S1400&SPINDX=SPINDX&Type=Type&NGlt=NGlt&startUserDefined=true&c1_val=&c2_val=&c3_val=&c4_val=&sort_attr=jname&sort_order=asc&condition=&pulsar_names=&ephemeris=short&coords_unit=raj/decj&radius=&coords_1=&coords_2=&style=Short+without+errors&no_value=*&fsize=3&x_axis=&x_scale=linear&y_axis=&y_scale=linear&state=query&table_bottom.x=40&table_bottom.y=24'
            Hurl2='#NAME|PSRJ|RAJ|DECJ|POSEPOCH|F0|F1|F2|PEPOCH|DM|W50|W10|S400|S1400|SPINDX|PSRTYPE|NGLT\n'
            sock = urllib.urlopen(url2)
            data = sock.read()
            sock.close()
            
# massage the data
            data = data.split('<pre>')[1]
            data = data.split('</pre>')[0]
            data = data.splitlines()[5:-1]

# save the data to a file
	    if save == True:
                
# save pulsar data to file and backup previous versions
                try:
		    f = open('pulsar_list.txt','w')
		    print>>f,Hurl2
		    for line in data:
			    print>>f,line
		    f.close()
		except:
                    pass

                idx = 0
		fbase = 'pulsar_list.txt.bz2'
		fout = fbase
		while os.path.exists(fout):
                    fout = fbase + '.' + str(idx)
		    idx += 1
		if os.path.exists(fbase):
		    os.system('mv %s %s' % (fbase, fout))
		os.system('bzip2  pulsar_list.txt') 
		    
            data = [data[i].split() for i in range(len(data)) if len(data[i]) > 1]


#names = ('indx','name','psrj','raj','decj','posepoch','f0','f1','f2',
#                     'pepoch','dm','w50','w10','s400','s1400','spindx')
        except:
            data = []



        srcs = __data2pulsarlist__(data=data)

        return srcs

def __data2pulsarlist__(data=[]):
    """
    Given an array of data,
    (with cols 'indx','name','psrj','raj','decj','posepoch',
    'f0','f1','f2','pepoch','dm','w50','w10','s400','s1400','spindx','psrtype','nglt')
    
    Return an array of pulsar objects:
    
    """

    srcs = []
    for po in data:
            name = po[1] #Use NAME (B-name if it exists PSRJ2000 name otherwise)
            raj = po[3]#ephem.hours(po[3])
            decj = po[4]#ephem.degrees(po[4])
            try:
                    #NOTE: The ATNF PosEpoch is *when* the coord was determined
                    #but the coords are still J2000 coords.
                epoch = np.float(po[5])
            except:
                epoch = ephem.J2000
            d=dict(indx=int(po[0]),name=po[1],psrj=po[2],f0=po[6],
                   f1=po[7],f2=po[8],pepoch=po[9],dm=po[10],
                   w50=po[11],w10=po[12],s400=po[13],s1400=po[14],
                   spindx=po[15],psrtype=po[16],nglt=po[17])
            
            srcs.append(pulsar(raj,decj,name=name,epoch=ephem.J2000,dct=d))


    return srcs
        

#               
#    ______  __ __|  |   ___________ _______ 
#    \____ \|  |  \  |  /  ___/\__  \\_  __ \
#    |  |_> >  |  /  |__\___ \  / __ \|  | \/
#    |   __/|____/|____/____  >(____  /__|   
#    |__|                   \/      \/       
#


class pulsar(a.amp.RadioFixedBody):
    """
    A class describing pulsars. We simply extend the aipy.amp.RadioFixedbody,
    though we add the spectral index and provide a pulsed-flux routine.
    
    Notes:
    jys = pulsar flux is read from S400, otherwise scaled down from S1400,
          otherwise = np.nan
	  We store the 400MHz flux
    spindx defaults to -0.7 if none defined in ATNF

    """


    def __init__(self,ra,dec,name='',epoch=ephem.J2000,dct={}):

        self.src_name = name
        self.props = dct
        mjys, spindx = self.flux_spindex() #Flx came in mJy, aipy likes Jy
        a.amp.RadioFixedBody.__init__(self,ra,dec,name=name,epoch=epoch,
				      mfreq=0.4,jys=mjys/1000.,index=spindx) 

#Convert all appropriate entries to floats and
#calculate P0 and P1, since we are only querying f0 and f1
        for key in self.props.keys():
            if key not in ['psrj','name','psrtype']:
                try:
                    self.props[key] = np.float(self.props[key])
                except:
                    self.props[key] = np.nan
				
	self.props['p0'] = 1./self.props['f0']
	try:
            self.props['p1'] = -self.props['p0']*self.props['f1']
	except:
            pass

	if np.isnan(self.props['p1']):
            self.props['p1'] = 0. 

	try:
            self.props['p2'] = -(self.props['p1'] - self.props['f2'])/self.props['f0']
	except:
            pass

#phase = F0*t + F1*t^2 + F2*t^3..., set up the np.polyval array to quickly
#calculate the phase
	phasev = [0]
	for key in ['f0','f1','f2']:
            val = self.props[key]  
            if not np.isnan(val):
                if key == 'f1': val = val/2.    #see eqn 120 of Hobbs,Edwards,Manchester (tempo2).
                elif key == 'f2': val = val/6.  #I include the factorial here
                phasev.append(np.float(self.props[key]))
	self.props['phasev'] = phasev[::-1]

        if np.isnan(self.props['p2']):
            self.props['p2'] = 0.

	self.name = self.props['name']

	if np.isnan(self.props['pepoch']):
            self.props['pepoch'] = ephem.J2000

#split off the references
        self.props['psrtype'] = self.props['psrtype'].split('[')[0]

#rank of the pulsars importance (use your own algs., they all start with equal weight, 1)
        self.weight = 1
#pphase: If None, use barycentric timing (we don't make any corrections).
#otherwise, call self.pregenerate_PulsePhase to populate this, giving accurate
#timing.
        self.pphase = None


    def flux_spindex(self):
        """
        Given the dictionary of pulsar properties, return the 
        flux at 400MHz and the spectral index.

        If ATNF has s400=s1400=0/*, we set flx=1 at 400MHz
        If no spectral index is defined, default to -0.7

        If s400 is not given, use s1400 and spindx to calc. s400
        """
#input properties
        ips = self.props
#output properties. These get updated if passed in 'props'
        ops = dict(spindx = -0.7, s400 = 1, s1400 = .73717)
        useprops = dict(spindx = False, s400 = False, s1400 = False)
        

#check for passed values, otherwise set spindex=-0.7, s400=10mJy
	if ips['spindx'] == '*':
           ips['spindx'] = ops['spindx']

	if ips['s400'] == '*' and ips['s1400'] == '*':
		ips['s400'] = ops['s400']

        for key in ops.keys():
            if ips.has_key(key):
                try:
                    ops[key] = np.float(ips[key])
                    useprops[key] = True
                except:
                    pass

        if useprops['s400']:
            return ops['s400'],ops['spindx']
        else:
            if useprops['s1400']:
                return ops['s1400']*(400./1400.)**(ops['spindx']),ops['spindx']
            else:
                return 0., ops['spindx']

    def get_pjys(self,t,freqs,Flx=None,DM=None,err=0):
        """
	Return the pulsed fluxes vs. freq, accounting for DM.
	Can optionally override the objects intrinsic flux.

	We generate a Gaussian random noise about the Flx and modulate it
	(Amplitude modulated noise) if phase(t) < W50, otherwise return 0.
	Can optionally add normally-distributed error to all parts of the 
	phase with sigma=sqrt(var)= err*Flx

	

	Args:
	t = JD
	freqs = list of active frequencies [GHz]
	Flx = override S400 flux, scaling to obs. freqs
         	according to intrinsic spindx
		Note: we add Gaussian random fluctuations to Flx,
		 which scales the overall pulse at each time step
	       

	DM = override pulsars value of DM
	err = noise added to signal (as fraction of Flx)
	

	Returns array of fluxes: flux(freqs)
	"""

#get Flx of object
	if Flx is None:
#note. self._jys = source strength at mfreq = 0.4GHz

            try:
                Flx = self.get_jys()  #flx(freq) scaled to active freqs.
		jys = normal(loc=0., scale=Flx )
	    except:
                self.update_jys(freqs)
		Flx = self.get_jys() #flx(freq) scaled to active freqs.
                jys = normal(loc=0., scale=Flx )
	else:
            spindx = self.props['spindx']
#recall, flux is defined at 400MHz
            Flx = Flx * (freqs/0.4)**spindx
#	    jys = np.ones_like(Flx)
	    jys = normal(loc=0., scale=Flx )

#get DM of object	    
	if DM is None:
            DM = self.props['dm']
	else:
            DM = 0.
#get DM offset of pulse (inf. freq arrives first)
	dt = 4.148808e-3 *DM /(freqs**2) #in [s], [freqs]=GHz
#	mfreq = np.mean(freqs)
#        dt = -4148.808 * (1./mfreq**2-1./freqs**2) * DM * 1.e-6



#if self.pphase, then we've used Ingrid's 'pphase' to make/parse real polyco's
#otherwise we simply use the barycentric ephemeris  
        t = (t - 2400000.5)  

        if type(self.pphase) == type(np.array((0))):
#self.pphase [[mjd phase f0 f1 f2]]         
#           if self.idx: #AARDEL   
#               idx = self.idx #np.abs((self.pphase[:,0] - t)).argmin()
#           else:
           idx = np.abs((self.pphase[:,0]-t)).argmin()
           phasev = self.pphase[idx,1:][::-1]
           pepoch = self.pphase[idx,0]
        else:
#note, this is the barycenter ephemeris. 
#you should really look into getting tempo, polyco, and pphase working
           phasev = self.props['phasev']
           pepoch = self.props['pepoch']
           

#number of days from obs. time 't' to recorded PEpoch
	t = t - pepoch #[days from PEPOCH]
	tees = t*86400. + dt #[sec]: dispersed arrival time as fnt of frequency

#recall phasev = [...,F2/6,F1/2.,F0,PHI0]
	phase = np.mod(np.polyval(phasev,tees),1.)
#AARdel        phase = np.polyval(phasev,tees)
#find out if pulsar is pulsed:
#use W50 pulse width, then w10, then 5% duty cycle #Note w50 and w10 are in ms
#	curP0 = self.props['p0'] #pulse period [s] (at PEpoch)
	curP0 = 1./np.polyval(phasev[:-1],t) #pulse period[s] (at 't')

	if not np.isnan(self.props['w50']):
            wdth = self.props['w50']/1000./curP0 #width (in Phase of pulse)
	elif not np.isnan(self.props['w10']):
            wdth = self.props['w10']/1000./curP0
	else:
            wdth = 0.1

	notpulsed = np.nonzero(phase > wdth)

#get noise of object: Gaussian random or zero 
	if err != 0:
            noise = normal(loc=0.,
			   scale=np.abs(err*jys)
			   )
#add noise to all parts of data, only signal to frequencies which are pulsing...
	    jys += noise
	else:
            noise = np.zeros_like(jys)
	    
	jys[notpulsed] = noise[notpulsed]
        
        return jys
	

    def pregenerate_PulsePhase(self,tstart,tend,freq=None,obs=None):
        """
	(pre)Generate the pulse phase of the pulsar using
	Ingrid's 'pphase' program. If generated, we use
	this data to determine the pulse phase in get_pjys
	
	Note:
	this program makes system calls to 'sigproc-4.3/polyco'
	and Ingrid's 'pphase' program and will fail if not found.

        tstart: [MJD] for start of observations
        tend: [MJD] for end of observations
        dt: [s] sampling time
        freq: [MHz] defaults to 1400 (tempo TZRFRQ)
        obs: tempo obsys.dat observatory code


	"""

#store the [mjd,phase,p0] info in self.pphase
#start with it undefined, but update it if polyco and pphase succeed
        self.pphase = None

#check in tempo dir for the par file
        try:
            tempodir = os.environ['TEMPO']
            tempocfg = np.genfromtxt(tempodir + '/tempo.cfg',dtype='str')
            pardir = ''
            for i in tempocfg:
                if i[0] == 'PARDIR': 
                    pardir = i[1]
            parname = self.name[1:] + '.par'
            parfile = pardir + parname
            if os.path.exists(parfile):
                pass
            else:
                raise Exception()
        except:
#otherwise create the file with basic information
            print "Using temporary par file for %s"\
                  % self.name
            self.createpar(freq,obs)
            parfile = self.parfile
            
#generate the polycos:
        cmd = ['polyco',self.name[1:],'-par',parfile,
               '-mjd',str(int(tstart)),
               '-maxha','12',
               '-nspan','60',
               '-mjds','%5.3f' % (tstart - 0.5),
               '-mjdf','%5.3f' % (tend + 1.5), 
               ]
        if freq:
           cmd.append('-freq')
           cmd.append(str(freq))
        if obs:
           cmd.append('-site')
           cmd.append(str(obs))
	fo = subprocess.Popen(cmd,stdout=subprocess.PIPE).communicate()[0]
#        os.unlink(self.parfile)

        if 'not found' in fo or 'Error' in fo:
            self.pphase = None
            print "\n\t polyco.dat generation failed:\n\t%s\n" % cmd
            print "%s will use barycentre ephemeris" % self.name
        else:
#generate pphase data for interval of interest
            try:
                cmd = ['./pphase',str(tstart),str(tend),str(.00144)] #note: pphase has 1e-6 precision in mjd = 0.00144 minutes
                pp = subprocess.Popen(cmd,stdout=subprocess.PIPE).communicate()[0]
                tmpfile = tempfile.TemporaryFile()
                tmpfile.writelines(pp)
                tmpfile.flush()
                tmpfile.seek(0)

                ppn2 = np.genfromtxt(tmpfile,dtype='5f8',comments='#')
                self.pphase = ppn2
                print "Successfully generated polycos for %s " % self.name
            except:
                self.pphase = None
                print "Bad results from cmd %s" % cmd
                print "Not generating phase data for %s" % self.name
        
        

    def createpar(self,freqs=None,obs=None):
         """
         Create a (temporary) TEMPO.par file for this object.
         
         obs: tempo obsys.dat observatory code (defaults to arecibo)

         """

         pfile = tempfile.NamedTemporaryFile(prefix=self.name+'_',
                                             suffix='.par',dir='/tmp/',
                                             delete=False)

         pfile.writelines('PSR\t %s\n' % self.name[1:])
         pfile.writelines('RA\t %s\n' % self._ra)#J2000
         pfile.writelines('DEC\t %s\n' % self._dec)
         
         keys = ['f0','f1','f2','pepoch','dm']
         props = {'f0':'F0','f1':'F1','f2':'F2',
                  'pepoch':'PEPOCH','dm':'DM'}
         for key in keys:
             if self.props.has_key(key):
                 if np.isnan(self.props[key]): 
                     continue
                 else:
                     pfile.writelines('%s \t %s\n' %
                                  (props[key],self.props[key]))

         pfile.writelines('EPHEM\t DE200\n')
         pfile.writelines('CLK\t UNCORR\n')
         if freqs:
             pfile.writelines('TZRFRQ\t %s\n' % freqs)
         if obs:
             pfile.writelines('TZRSITE\t %s\n' % obs)
         pfile.writelines('# Produced by pulsars.py\n')
         pfile.close()

         self.parfile = pfile.name
        


#
#   _________              _________         __         .__                 
#  /   _____/______   ____ \_   ___ \_____ _/  |______  |  |   ____   ____  
#  \_____  \\_  __ \_/ ___\/    \  \/\__  \\   __\__  \ |  |  /  _ \ / ___\ 
#  /        \|  | \/\  \___\     \____/ __ \|  |  / __ \|  |_(  <_> ) /_/  >
# /_______  /|__|    \___  >\______  (____  /__| (____  /____/\____/\___  / 
#         \/             \/        \/     \/          \/           /_____/  
#  

class PSrcCatalog(a.amp.SrcCatalog):
    """
    Class for holding a catalog of celestial sources.
    Adds get_pjys method in order to get pulsed signals

    """
#    def __init__(self,*srcs,**kwargs):
#        a.amp.SrcCatalog(self,*srcs,**kwargs)

    def get_pjys(self,t,freqs,Flx=None,DM=None,err=0,srcs=None):
        """Return the list of (pulsed) fluxes of all src objects in catalog.
           Note, this routine accommodates non-pulsed sources too"""
        if srcs is None: srcs = self.keys()
        a = []
        for s in srcs:
            if type(self[s]).__name__ == 'pulsar':
                a.append(self[s].get_pjys(t,freqs,Flx=Flx,DM=DM,err=err))
            else:
                a.append(self[s].get_jys())

        return np.array(a)

    def get_jys(self,t,freqs,Flx=None,DM=None,err=0,srcs=None):
        """Return the list of pulsed and non-pulsed fluxes of all src objects in catalog"""

        if srcs is None: srcs = self.keys()
        a = []
        for s in srcs:
            if type(self[s]).__name__ == 'pulsar':
                a.append(self[s].get_pjys(t,freqs,Flx=Flx,DM=DM,err=err))
            else:
                a.append(self[s].get_jys())
        return np.array(a)

	
    def get_pjysOld(self,t,freqs,Flx=None,DM=None,err=1.e-10,srcs=None):
        """Return the list of (pulsed) fluxes of all src objects in catalog"""
        if srcs is None: srcs = self.keys()
        return np.array([self[s].get_pjysOld(t,freqs,Flx=Flx,DM=DM,err=err) for s in srcs])

	
	
class SrcCatalog(dict):
    """A catalog of celestial sources.  Can be initialized with a list
    of src objects, of as an empty catalog."""
    def __init__(self, *srcs, **kwargs):
            dict.__init__(self)
            self.add_srcs(*srcs)
    def add_srcs(self, *srcs):
            """Add src object(s) (RadioFixedBody,RadioSpecial) to catalog."""
            if len(srcs) == 1 and getattr(srcs[0], 'src_name', None) == None:
                    srcs = srcs[0]
                    for s in srcs: self[s.src_name] = s
    def get_srcs(self, *srcs):
        """Return list of all src objects in catalog."""
        if len(srcs) == 0: srcs = self.keys()
        elif len(srcs) == 1 and type(srcs[0]) != str:
            return [self[s] for s in srcs[0]]
        else: return [self[s] for s in srcs]
    def compute(self, observer):
        """Call compute method of all objects in catalog."""
        for s in self: self[s].compute(observer)
