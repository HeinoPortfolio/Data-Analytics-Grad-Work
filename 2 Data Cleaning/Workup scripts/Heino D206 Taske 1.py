# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:53:46 2023

@author: mehei
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 23:19:34 2023

@author: Matthew Heino

"""
#import fancyimpute as fi
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns

from fancyimpute import KNN



# Import the Medical data set.
medical_df = pd.read_csv('medical_raw_data.csv') 
print(medical_df.head(5))
print(medical_df.dtypes) 
print(medical_df.shape)

""""
Step 1: Remove Duplicate or unneeded elements from the dataframe
    
    Actions:
        
        1) Drop the first row of the datset.  Information is redundant and 
        not needed.
        
        2) Look for duplicates in each column of the dataframe.

"""

medical_df.drop(medical_df.columns[0], axis=1, inplace = True)

# Check to see that the columns was successfully dropped.
print(medical_df.head(5)) 
print(medical_df.shape)


"""
Step 2:  Fix formatting and inconsistencies in the data.

    Actions:
        
        1) Look for unique categories in the each of the categorical columns 
        of the dataframe.
        
        2) Reduce  the number of categories where appropriate.
    
        3) Change the data type of columns to a more appropriate type.
    

"""

# Check for uniqueness among the columns. Will check all columns with type 
# object as they may be categorical and if catergorical see if there are too 
# many categories.

# Check unique Customer IDs)
#print("\nUnique Customer IDs: ", medical_df.Customer_id.unique().size)
# Only unique customer IDs.


# Check unique Interactions.
#print("\nUnique Interactions: ", medical_df.Interaction.unique().size)
# Only unique Interactions.

# Check unique UIDs.
#print("\nUnique UIDs: ", medical_df.UID.unique().size)
# Only unique UIDs.


# City will be skipped 
# Diffcult to for unique values 
# Check City for uniqueness
#print("\nUnique cities",np.sort(medical_df.City.unique()))
#print("\nUnique Cities size: ", medical_df.City.unique().size)

 
# Check state for unique values 
#print("\nUnique states",np.sort(medical_df.State.unique()))
#print("\nUnique states size: ", medical_df.State.unique().size)
# only 52 unique


# County will be skipped 
# Diffcult to for unique values 
# Check County for uniqueness and valid entries

#print("\nUnique Counties",np.sort(medical_df.County.unique()))
#print("\nUnique Counties size: ", medical_df.County.unique().size)

""" **************************************************************************
    
    ZIP code checked for right length.
    This will show an error since this is currently stored as an integer
    This error will be noted in the assessment document in greater detail.
    Note this corresponds to Section D2 of the included document.

******************************************************************************
"""

#print("Minimum: ", medical_df['Zip'].min())
# Note this 
# Convert Zipcode from int64 to a String to get back leading zero in Zipcode.
medical_df = medical_df.astype({'Zip': str})

# Check data types of the columns.
#print(medical_df.Zip.dtypes)

# Check the Zip code to appropriate length.
# Does not return appropriate length.
#print(medical_df.Zip.str.len().min())

# Add appropriate amount of zeroes to the left of the Zip string.

medical_df['Zip'] = medical_df['Zip'].str.zfill(5)
#print("Min Zip: ",medical_df.Zip.str.len().min())
#print("Max Zip: ",medical_df.Zip.str.len().max())
#print(medical_df['Zip'].str.startswith('0'))
#print("Zip Type: ", medical_df.Zip.dtypes)
#print(medical_df.dtypes)    
     
#Check Area for uniqueness

#print("\nUnique values of Area: ",np.sort(medical_df.Area.unique()))     
# All values are acceptable      
      
# Check Timezone
# Will undergo category collapsing.


""" **************************************************************************
    
    Tmezone checked for the amount of timezone categories.
    Found to have 26 will be reduced to 7 timezones.  
    Note this corresponds to Section D2 of the included document.

******************************************************************************
"""
#print(np.sort(medical_df.Timezone.unique())) 
#print(medical_df.Timezone.unique().size)  


timezone_replace_dict = {'America/Adak': 'Hawaii-Aleutian', 
                         'America/Anchorage' : 'Alaska',
                         'America/Boise' : 'Mountain',
                         'America/Chicago' :'Central',
                         'America/Denver' :  'Mountain',
                         'America/Detroit' :'Eastern', 
                         'America/Indiana/Indianapolis' : 'Eastern',
                         'America/Indiana/Knox' : 'Central', 
                         'America/Indiana/Marengo' : 'Eastern',
                         'America/Indiana/Tell_City' : 'Central',
                         'America/Indiana/Vevay' : 'Eastern',
                         'America/Indiana/Vincennes' : 'Eastern',
                         'America/Indiana/Winamac' : 'Eastern',
                         'America/Kentucky/Louisville' : 'Eastern', 
                         'America/Los_Angeles' : 'Pacific',
                         'America/Menominee' : 'Central',
                         'America/New_York': 'Eastern',
                         'America/Nome' : 'Alaska',
                         'America/North_Dakota/Beulah' : 'Central',
                         'America/North_Dakota/New_Salem' : 'Central',
                         'America/Phoenix' : 'Mountain',
                         'America/Puerto_Rico' : 'Atlantic',
                         'America/Sitka' : 'Alaska',
                         'America/Toronto' : 'Eastern', 
                         'America/Yakutat' : 'Alaska', 
                         'Pacific/Honolulu' : 'Hawaii-Aleutian'
    }

medical_df['Timezone'].replace(timezone_replace_dict, inplace=True)    
#print(np.sort(medical_df.Timezone.unique())) 
 
#Check Job
#print("\n Jobs: ",np.sort(medical_df.Job.unique()))
print(medical_df.Job.unique().size) 



# Replace 
medical_df['Job'].replace("^Engineer.*","Engineer", regex=True, inplace=True)
medical_df['Job'].replace("^Civil Engineer.*","Engineer", regex=True, inplace=True)
medical_df['Job'].replace("^Civil engineer.*","Engineer", regex=True, inplace=True)

medical_df['Job'].replace("^Teacher.*","Education", regex=True, inplace=True)
medical_df['Job'].replace("teacher$","Education", regex=True, inplace=True)
medical_df['Job'].replace(".*?Education$","Education", regex=True, inplace=True)
medical_df['Job'].replace(".*?education$","Education", regex=True, inplace=True)
medical_df['Job'].replace(".*?lecturer$","Education", regex=True, inplace=True)
medical_df['Job'].replace("^Education.*","Education", regex=True, inplace=True)

edu_dict = {'Music tutor': 'Education',  'Learning mentor':'Education', 
            'Professor Emeritus' :'Education',
            'Higher education careers adviser':'Education'    
            }
medical_df['Job'].replace(edu_dict, inplace=True)


medical_df['Job'].replace("^Accountant.*","Accountant", regex=True, inplace=True)
medical_df['Job'].replace(".*?accountant$","Accountant", regex=True, inplace=True)

medical_df['Job'].replace("^Finan.*","Finance", regex=True, inplace=True)
medical_df['Job'].replace("^Tax.*","Finance", regex=True, inplace=True)
medical_df['Job'].replace(".*?banker$","Finance", regex=True, inplace=True)
medical_df['Job'].replace("^Investment.*","Finance", regex=True, inplace=True)

fin_dict ={'Senior tax professional/tax inspector' :'Finance',
           'Risk analyst' :'Finance', 'Loss adjuster, chartered' :'Finance',
           'Insurance claims handler' : 'Finance','Futures trader' :'Finance',
           'Equities trader': 'Finance', 'Bonds trader' :'Finance'
    }
medical_df['Job'].replace(fin_dict, inplace=True)

medical_df['Job'].replace("^Secretary.*","Office", regex=True, inplace=True)
medical_df['Job'].replace(".*?secretary$","Office", regex=True, inplace=True)

medical_df['Job'].replace(".*?manager$","Management", regex=True, inplace=True)

medical_df['Job'].replace("^Administrator.*","Administration", regex=True, inplace=True)
medical_df['Job'].replace(".*?administrator$","Administration", regex=True, inplace=True)

medical_df['Job'].replace("^Chief.*","Executive", regex=True, inplace=True)


medical_df['Job'].replace("^Research.*","Research", regex=True, inplace=True)
medical_df['Job'].replace(".*?researcher$","Research", regex=True, inplace=True)

medical_df['Job'].replace("^Scientist.*","Scientist", regex=True, inplace=True)
medical_df['Job'].replace(".*?scientist$","Scientist", regex=True, inplace=True)
medical_df['Job'].replace(".*?physicist$","Scientist", regex=True, inplace=True)
medical_df['Job'].replace(".*?biologist$","Scientist", regex=True, inplace=True)
medical_df['Job'].replace(".*?geologist$","Scientist", regex=True, inplace=True)
medical_df['Job'].replace("^Geo.*","Scientist", regex=True, inplace=True)

sci_dict = {'Seismic interpreter':'Scientist', 'Plant breeder/geneticist': 'Scientist',
        'Meteorologist' : 'Scientist','Cytogeneticist':'Scientist',
		'Metallurgist' :'Scientist','Physicist, medical' :'Scientist',
        'Scientific laboratory technician' :'Scientist', 
        'Teaching laboratory technician' : 'Scientist', 'Hydrologist' :'Scientist',
        'Herpetologist' :'Scientist' ,'Geneticist, molecular' :'Scientist'
    }

medical_df['Job'].replace(sci_dict, inplace=True)

medical_df['Job'].replace("^Therapist.*","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace(".*?therapist$","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace("^Psychotherapist.*","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace("^Psychologist.*","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace(".*?psychologist$","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace("^Nurse.*","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace(".*?nurse$","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace("^Clinical.*","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace(".*?clinical$","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace("^Surgeon.*","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace("^Toxicologist.*","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace(".*?radiographer$","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace("^Pharmacist.*","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace("^Opt.*","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace(".*?doctor$","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace("^Pathologist.*","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace("^Pharmacologist.*","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace("^Podiatrist.*","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace("^Immunologist'.*","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace("^Chiro.*","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace("^Doctor.*","Healthcare", regex=True, inplace=True)
medical_df['Job'].replace("^Health.*","Healthcare", regex=True, inplace=True)

medical_dict ={'Neurosurgeon':"Healthcare",  'Occupational hygienist':"Healthcare",
 'Oceanographer': "Healthcare", 'Oncologist':"Healthcare", 'Acupuncturist':"Healthcare",
 'Ophthalmologist' :"Healthcare", 'Orthoptist':"Healthcare",'Osteopath': "Healthcare" ,
 'Paramedic':"Healthcare",'Midwife':"Healthcare",'Exercise physiologist':"Healthcare",
 'Homeopath':"Healthcare",'Immunologist':"Healthcare", 'Psychiatrist': "Healthcare",
 'Veterinary surgeon' : "Healthcare", 'Special effects artist': "Healthcare",
  'Herbalist': "Healthcare", 'Haematologist' :'Healthcare',
  'Environmental health practitioner':'Healthcare'
 }


medical_df['Job'].replace(medical_dict, inplace=True)

medical_df['Job'].replace("^Surveyor.*","Surveyor", regex=True, inplace=True)
medical_df['Job'].replace(".*?surveyor$","Surveyor", regex=True, inplace=True)
medical_df['Job'].replace("^Engineer.*","Engineer", regex=True, inplace=True)
medical_df['Job'].replace(".*?engineer$","Engineer", regex=True, inplace=True)

medical_df['Job'].replace("^Legal.*","Legal", regex=True, inplace=True)
medical_df['Job'].replace(".*?attorney$","Legal", regex=True, inplace=True)
medical_df['Job'].replace("^Lawyer.*","Legal", regex=True, inplace=True)
medical_df['Job'].replace("^Solicitor.*","Legal", regex=True, inplace=True)
medical_df['Job'].replace("^Barrister.*","Legal", regex=True, inplace=True)

medical_df['Job'].replace("^Televis.*","Entertainment", regex=True, inplace=True)
medical_df['Job'].replace("^Production.*","Entertainment", regex=True, inplace=True)
medical_df['Job'].replace("^Producer.*","Entertainment", regex=True, inplace=True)
medical_df['Job'].replace("^Broadcasting.*","Entertainment", regex=True, inplace=True)
medical_df['Job'].replace(".*?broadcasting$","Entertainment", regex=True, inplace=True)
medical_df['Job'].replace(".*?video$","Entertainment", regex=True, inplace=True)
medical_df['Job'].replace("^Radio.*","Entertainment", regex=True, inplace=True)
medical_df['Job'].replace("^Theatre.*","Entertainment", regex=True, inplace=True)

ent_dict={'Musician' : 'Entertainment', 'Video editor': 'Entertainment',
          'Multimedia specialist': 'Entertainment',
          'Film/video editor' :'Entertainment','Best boy': 'Entertainment',
          'Broadcast presenter': 'Entertainment', 'Fine artist' :'Entertainment',
          'Event organiser': 'Entertainment', 'Gaffer':'Entertainment'
          }

medical_df['Job'].replace(ent_dict, inplace=True)

medical_df['Job'].replace("^Armed.*","Military", regex=True, inplace=True)

medical_df['Job'].replace("^Sales.*","Sales", regex=True, inplace=True)
medical_df['Job'].replace(".*?broker$","Sales", regex=True, inplace=True)

medical_df['Job'].replace("^Retail.*","Retail", regex=True, inplace=True)
medical_df['Job'].replace("^Barista.*","Retail", regex=True, inplace=True)
medical_df['Job'].replace(".*?retail$","Retail", regex=True, inplace=True)
medical_df['Job'].replace(".*?buyer$","Retail", regex=True, inplace=True)

medical_df['Job'].replace(".*?consultant$","Consultant", regex=True, inplace=True)

medical_df['Job'].replace("^Designer.*","Designer", regex=True, inplace=True)
medical_df['Job'].replace(".*?designer$","Designer", regex=True, inplace=True)

medical_df['Job'].replace("^Journalist.*","Journalism", regex=True, inplace=True)
medical_df['Job'].replace("^Magazine.*","Journalism", regex=True, inplace=True)
medical_df['Job'].replace("^Press.*","Journalism", regex=True, inplace=True)
medical_df['Job'].replace(".*?journalist$","Journalism", regex=True, inplace=True)
medical_df['Job'].replace(".*?writer$","Journalism", regex=True, inplace=True)
medical_df['Job'].replace(".*?author$","Journalism", regex=True, inplace=True)

medical_df['Job'].replace("^Museum.*","Museum", regex=True, inplace=True)
medical_df['Job'].replace(".*?gallery$","Museum", regex=True, inplace=True)

medical_df['Job'].replace("^Advertising .*","Advertising", regex=True, inplace=True)

it_dict = {'Systems analyst':"IT", 'Systems developer': "IT",
           'Multimedia programmer' :"IT" }
medical_df['Job'].replace("^Programmer.*","IT", regex=True, inplace=True)
medical_df['Job'].replace(it_dict, inplace=True)

medical_df['Job'].replace("^Public relations.*","Public Relations", regex=True, inplace=True)

medical_df['Job'].replace(".*?government$","Government", regex=True, inplace=True)
medical_df['Job'].replace(".*?officer$","Government", regex=True, inplace=True)
medical_df['Job'].replace("^Ranger/warden.*","Government", regex=True, inplace=True)

medical_df['Job'].replace("^Architect.*","Architect", regex=True, inplace=True)
medical_df['Job'].replace(".*?architect$","Architect", regex=True, inplace=True)

misc_dict={ 'Publishing copy' :'Misc','Proofreader':'Misc','Statistician' :'Misc', 
    'Sub' :'Misc','Sports coach':'Misc','Printmaker':'Misc',
    'Lexicographer':'Misc','Licensed conveyancer': 'Misc',
    'Transport planner': 'Misc','Technical brewer': 'Misc',
    'Radiation protection practitioner': 'Misc','Brewing technologist': 'Misc',
    "Politician's assistant":'Misc', 'Translator': 'Misc',
    'Patent examiner':'Misc', 'Personal assistant':'Misc',
    'Photographer':'Misc','Lobbyist':'Misc','Social worker':'Misc'
   }
medical_df['Job'].replace(misc_dict, inplace=True)


avi_dict={'Pilot, airline' : 'Aviation', }
medical_df['Job'].replace(avi_dict, inplace=True)

hort_dict ={'Tree surgeon' : 'Horticulture'}
medical_df['Job'].replace(hort_dict, inplace=True)

print("After replacement",medical_df.Job.unique().size) 
print("\n Jobs: ",np.sort(medical_df.Job.unique()))
print(medical_df.Job.unique().size) 
print(medical_df.shape)
print(medical_df['Job'].head(5))





# Section D5 of the document ************************************************* 
# write the dataframe to a file
#medical_df.to_csv('cleaned_csv.csv') 
      
      
      
