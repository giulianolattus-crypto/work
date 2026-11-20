
# coding: utf-8

# In[9]:

#This script takes a initial date and final date and downloads the files needed 
#to compute the wave activity flux (WAF)
#for that given period. NCEP/NCAR Reanalysis (Kalnay etal 1996)are used 

#cambios: uso de la funcion corriente en vez de hgt. nivel 0.21 sigma

# As an example to run by shell: python Calculo_WAF.py --dateinit "2018-02-01" --dateend "2018-02-28"

#libraries needed
import urllib.request
from bs4 import BeautifulSoup
import netCDF4
import numpy as np
import argparse 
import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs	
import numpy as np 
import cartopy.feature 	
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import copy
import os
from numpy import ma  #mask smaller values       

# In[]      
def clean():   #clean enviroment
    os.system("rm -f /tmp/*.gz /tmp/*.nc")
# In[]      Downloads netcdf from esrl    
def descarga_nc( mesi, diai, mesf, diaf, aniof, variable_entrada, variable_salida, tipo):
    #tipo controls type of data: 2 for anomalies, 3 for climatology
    # Open NCEP NCAR to access to lik to data
        url = 'http://www.psl.noaa.gov/cgi-bin/data/composites/comp.day.pl?var='+variable_entrada+'&level=.2101+sigma&iy[1]=&im[1]=&id[1]=&iy[2]=&im[2]=&id[2]=&iy[3]=&im[3]=&id[3]=&iy[4]=&im[4]=&id[4]=&iy[5]=&im[5]=&id[5]=&iy[6]=&im[6]=&id[6]=&iy[7]=&im[7]=&id[7]=&iy[8]=&im[8]=&id[8]=&iy[9]=&im[9]=&id[9]=&iy[10]=&im[10]=&id[10]=&iy[11]=&im[11]=&id[11]=&iy[12]=&im[12]=&id[12]=&iy[13]=&im[13]=&id[13]=&iy[14]=&im[14]=&id[14]=&iy[15]=&im[15]=&id[15]=&iy[16]=&im[16]=&id[16]=&iy[17]=&im[17]=&id[17]=&iy[18]=&im[18]=&id[18]=&iy[19]=&im[19]=&id[19]=&iy[20]=&im[20]=&id[20]=&monr1='+str(mesi)+'&dayr1='+str(diai)+'&monr2='+str(mesf)+'&dayr2='+str(diaf)+'&iyr[1]='+str(aniof)+'&filenamein=&plotlabel=&lag=0&labelc=Color&labels=Shaded&type='+str(tipo)+'&scale=&label=0&cint=&lowr=&highr=&istate=0&proj=ALL&xlat1=&xlat2=&xlon1=&xlon2=&custproj=Cylindrical+Equidistant&level1=1000mb&level2=10mb&Submit=Create+Plot'
        
        response = urllib.request.urlopen(url)
        
        data = response.read()      # a `bytes` object

        soup = BeautifulSoup(data,'html.parser') #is an xml, beautifull has a module to manage it
        
        link = soup.findAll('img')[-1]['src']
        
        #A very inefficient way to get the nc file
        link=list(link)
        link[-3]='n'
        link[-2]='c'
        link[-1]=''
        
        #get nc file save as netcdf
        ruta = "./tmp/"
        if not os.path.exists(ruta):
            os.mkdir(ruta)

        urllib.request.urlretrieve('http://www.psl.noaa.gov'+"".join(link), ruta+variable_salida+'.nc')
# In[]      extract variable from netcdf
def manipular_nc(archivo,variable):

    dataset = netCDF4.Dataset(archivo, 'r')
    var_out = dataset.variables[variable][:]
    lon = dataset.variables['lon'][:]
    lat = dataset.variables['lat'][:]
    dataset.close()
    return var_out, lon, lat

# In[]  computation of derivatives.
        
def c_diff(arr, h, dim, cyclic = False):  #compute derivate of array variable respect to h associated to dim
    #adapted from kuchaale script
    ndim = arr.ndim
    lst = [i for i in range(ndim)]

    lst[dim], lst[0] = lst[0], lst[dim]
    rank = lst 
    arr = np.transpose(arr, tuple(rank))

    if ndim == 3:
        shp = (arr.shape[0]-2,1,1)
    elif ndim == 4:
        shp = (arr.shape[0]-2,1,1,1)
    
    d_arr = np.copy(arr)
    if not cyclic:  
        d_arr[0,...] = (arr[1,...]-arr[0,...])/(h[1]-h[0])
        d_arr[-1,...] = (arr[-1,...]-arr[-2,...])/(h[-1]-h[-2])
        d_arr[1:-1,...] = (arr[2:,...]-arr[0:-2,...])/np.reshape(h[2:]-h[0:-2], shp)

    elif cyclic:
        d_arr[0,...] = (arr[1,...]-arr[-1,...])/(h[1]-h[-1])
        d_arr[-1,...] = (arr[0,...]-arr[-2,...])/(h[0]-h[-2])
        d_arr[1:-1,...] = (arr[2:,...]-arr[0:-2,...])/np.reshape(h[2:]-h[0:-2], shp)

    d_arr = np.transpose(d_arr, tuple(rank))

    return d_arr
        
# In[] main code: parser data and code        
def main():
  
	# Define parser data
    parser = argparse.ArgumentParser(description='Plotting WAF for given dates')
    # First arguments. Initial Date Format yyyy-mm-dd
    parser.add_argument('--dateinit',dest='date_init', metavar='Date', type=str,
                        nargs=1,help='Initial date in "YYYY-MM-DD"')    
    #Second argument: Final Date Format yyyy-mm-dd
    parser.add_argument('--dateend',dest='date_end', metavar='Date', type=str,
                        nargs=1,help='Final date in "YYYY-MM-DD"')
    # Specify models to exclude from command line
    #parser.add_argument('--no-cfs', dest='cfs_bool', action="store_true", default= False, help="Don't display CFS V2 information")
    
    # Extract dates from args
    args=parser.parse_args()

    initialDate = datetime.datetime.strptime(args.date_init[0], '%Y-%m-%d')
    finalDate = datetime.datetime.strptime(args.date_end[0], '%Y-%m-%d')
    
    clean()


    #initial date
    iniy = initialDate.year
    inim = initialDate.month
    inid = initialDate.day
    
    #final date
    finy = finalDate.year
    finm = finalDate.month
    find = finalDate.day
    
    #download: psi anomalies and  climatology for the selected period
    
    variables = 'Streamfunction'
    
    out_var = ['strfc_climo','strfc']
    
    descarga_nc(inim,inid,finm,find,finy,variables,out_var[0],3)
    
    descarga_nc(inim,inid,finm,find,finy,variables,out_var[1],2)
                
    
    # manipulation of netCDF files
    ruta = "./tmp/"
    
    nc_var = 'psi'
    
    [psiclm,lon,lat] = manipular_nc(ruta+out_var[0]+'.nc',nc_var)
    
    [psiaa,lonpsi,latpsi] = manipular_nc(ruta+out_var[1]+'.nc',nc_var)  #psia [1 nlat nlon]
           
    # In[11]: Compute plumb fluxes. Script adapted from Kazuaki Nishii and Hisashi Nakamura
           
    #begin
    [xxx,nlats,nlons] = psiaa.shape #get dimensions

    #earth radius
    a = 6400000
    
    coslat = np.cos(lat*3.14/180)
    
    #climatological wind at psi level
    
    dpsiclmdlon = c_diff(psiclm,lon,2)
    
    dpsiclmdlat = c_diff(psiclm,lat,1)
    
    uclm = -1*dpsiclmdlat
    
    vclm = dpsiclmdlon
    
    magU = np.sqrt(np.add(np.power(uclm,2),np.power(vclm,2)))
    
    
    dpsidlon = c_diff(psiaa,lon,2)
    
    ddpsidlonlon = c_diff(dpsidlon,lon,2)
    
    dpsidlat = c_diff(psiaa,lat,1)
    ddpsidlatlat = c_diff(dpsidlat,lat,1)
    
    ddpsidlatlon = c_diff(dpsidlat,lon,2)
    
    termxu = dpsidlon*dpsidlon-psiaa*ddpsidlonlon
    
    termxv = dpsidlon*dpsidlat-ddpsidlatlon*psiaa
    
    termyv = dpsidlat*dpsidlat-psiaa*ddpsidlatlat
    
    # 0.2101 is the scale of p
    coeff1 = np.transpose(np.tile(coslat,(nlons,1)))*(0.2101)/(2*magU)
    
    #x-component
    px = coeff1/(a*a*np.transpose(np.tile(coslat,(nlons,1))))*( uclm*termxu/np.transpose(np.tile(coslat,(nlons,1))) + vclm*termxv)
    
    #y-component
    py = coeff1/(a*a)*( uclm/np.transpose(np.tile(coslat,(nlons,1)))*termxv + vclm*termyv)
    
#========================================================================================    
    # In[12]:plot flux and psi anomalies
    # create figure, add axes

    fig1 = plt.figure(figsize=(16, 11)) 
    
    ax = plt.subplot(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    #Pasamos las latitudes/longitudes del dataset a una reticula para graficar
    lons, lats = np.meshgrid(lon,lat)
    clevs = np.arange(-2.5e7,2.75e7,0.25e7)
    cmap = copy.copy(plt.cm.get_cmap("RdBu")) 
    
    ax.set_extent([0, 359, -88, 10], crs=crs_latlon)
    im=ax.contourf(lons, lats, psiaa[0, :, :], clevs, transform=crs_latlon, cmap=cmap, extend='both')
    
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(cmap.N-1))
    # add colorbar
    #cbar = plt.colorbar(im, fraction=0.052, pad=0.04,shrink=0.7,aspect=12)#,"right")
    cbar = plt.colorbar(im, fraction=0.052, pad=0.04,shrink=0.3,aspect=12)#,"right")
    #legend
    cbar.set_label('$m^{2}/s$')    

    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.set_xticks(np.linspace(0, 360,  7), crs=crs_latlon)
    ax.set_yticks(np.linspace(-80, 10,  10), crs=crs_latlon)      
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
   
    #contoour levels
    ax.contour(lons, lats, psiaa[0, :, :], clevs, colors='k', linewidths=0.5, transform=crs_latlon)
       #print title
    ax.set_title('Anomalías Función Corriente 0.2101 sigma '+str(inid)+'/'+str(inim)+'/'+str(iniy)+'-'+str(find)+'/'+str(finm)+'/'+str(finy))
    
    #save figure - tight option adjuts paper size to figure
    fig1.savefig('psi_'+'{:02d}'.format(inid)+'{:02d}'.format(inim)+str(iniy)+'-'+'{:02d}'.format(find)+'{:02d}'.format(finm)+str(finy)+'.png',dpi=300,bbox_inches='tight',orientation='landscape')
    
    #plot plumb fluxes and save again
    #mask wind data to only show the 40% stronger fluxes.
    Q60=np.percentile(np.sqrt(np.add(np.power(px,2),np.power(py,2))),60) 
    M = np.sqrt(np.add(np.power(px,2),np.power(py,2))) < Q60
    #mask array
    px_mask = ma.array(px,mask = M)
    py_mask = ma.array(py,mask = M)
    #plot vectors
    ax.quiver(lons[2:-1:2,2:-1:2], lats[2:-1:2,2:-1:2], px_mask[0,2:-1:2,2:-1:2],
              py_mask[0,2:-1:2,2:-1:2], width=1e-3, headwidth=3,#headwidht (default3)
                       headlength=2.2, transform=crs_latlon)  # (default5))	      
    ax.set_title('Anomalías Función Corriente 0.2101 sigma y Flujos de Plumb '+str(inid)+'/'+str(inim)+'/'+str(iniy)+'-'+str(find)+'/'+str(finm)+'/'+str(finy))
    
    #save figure
    fig1.savefig('psi_plumb_'+'{:02d}'.format(inid)+'{:02d}'.format(inim)+str(iniy)+'-'+'{:02d}'.format(find)+'{:02d}'.format(finm)+str(finy)+'.png',dpi=300,bbox_inches='tight',orientation='landscape')
    

# In[] 
    
#begin        
if __name__ == "__main__":
    main() 


