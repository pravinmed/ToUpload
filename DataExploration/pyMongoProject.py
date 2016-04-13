# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:40:52 2015

@author: praveenkumar
"""

import xml.etree.cElementTree as ET
import pprint
import re
import codecs
import json
from pymongo import MongoClient
from datetime import datetime
from xml.etree.ElementTree import ParseError

CREATED = [ "version", "changeset", "timestamp", "user", "uid"]

#County which are not in the given locations.
OUTSIDE_STATE=["Witchita","Powhatan","Petrosino","La Aljorra","Gercüş/Batman"]

#Valid postal code in the given locations
POSTAL_CODE=["94514", "94551", "95377", "95391","94568","94505", "94513", "94514","94506",
             "94513", "94517", "94582","95215", "95230", "95236","94574","94566","95391",
             "95377","95304", "95377","94588","94566","94552", "94566", "94586", "94588",
             "94544", "94552", "94586","94550","94506", "94513", "94517", "94551", "94582",
             "95140","94514", "95377", "95391","95313", "95360"]

#Validation regex for the  values in the tag
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\.()\t\r\n]')

#Validation regex for the Invalid tags in the xml 
problemcharsInTag = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')



#Get the document in the JSON format  for each node.
def shape_element(element):
    node = {}
    try:
        if element.tag == "node" or element.tag == "way" :
            print element.attrib
            id =  element.get('id')
            node['id'] = id
            node['type']=element.tag
            node['visible'] = element.get('visible')
            # for the position coordinates
            if element.tag == 'node':
                pos=[]
                pos.append(float(element.get('lat')))
                pos.append(float(element.get('lon')))
                node['pos'] = pos
            created={}
            # for the creator tag 
            for create in CREATED:
                created[create] = element.get(create)
            node['created']=created
            address = {}
            node_refs=[]
            whichway={}
            # the child tags
            for child in element:
                refVal = child.get('ref')
                if refVal is not None:
                    node_refs.append(refVal)
                kVal = child.get('k')
                if kVal is None:
                    continue
                problem = problemcharsInTag.search(kVal)
                if problem >0:
                    continue
               
                val = child.get('v')
                if kVal.find('highway') != -1:
                    whichway['source']='highway'
                    whichway['type'] = val
                if kVal.find('waterway')!= -1:
                    whichway['source'] ='waterway'
                    whichway['type'] = val
                if kVal.find('railway')!= -1:
                    whichway['source']='railway'
                    whichway['type'] = val
                if kVal.find('landuse')!= -1:
                    whichway['source']=kVal
                    whichway['type'] = val
                # Audit the data
                if kVal == 'name' and 'name' not in whichway:
                    prob = problemchars.search(val)
                    if prob >0:
                        newVal = val[:prob.start()]
                        whichway['name'] = newVal
                    else:
                        whichway['name']=val
                if kVal == 'name:en':
                    whichway['name'] = val
                
                if kVal.find('county') !=-1:
                    node['county'] = val
                        
                #audit of the State
                if kVal == 'addr:state' and val != 'CA':
                    address = {}
                    whichway={}
                    break
                else:
                    address['state'] = 'CA'
    
                
                #audit of the city
                if kVal == 'addr:city':
                    if val in OUTSIDE_STATE:
                        address = {}
                        whichway={}
                        break
                    else:
                        address['city'] = val
                        
                #audit of the postal code.
                if kVal=='addr:postcode':
                    if val not in POSTAL_CODE: 
                        address ={}
                        whichway={}
                        break
                #end of auditing thr data
                if kVal.find('housenumber') != -1:
                    address['housenumber']=(val)
                if kVal == 'addr:street':
                    address['street'] = val
                if kVal.find('postcode')!=-1:
                    address['postcode'] = val
                if kVal.find('amenity')!=-1:
                    node['amenity'] =val
                if kVal.find('religion')!= -1:
                    node['religion'] = val
                if kVal == 'name':
                    node['name']=val
                if kVal == 'cuisine':
                    node['cuisine']=val
                if kVal == 'phone':
                    node['phone']=val
                    
            #insert to the collection
            if bool(address):
                node['address']=address  
            if bool(whichway):
                node['pathtype']=whichway
            if len(node_refs) >0:        
                node['node_refs'] = node_refs 
            print node
            return node
        else:
            return None
    except ParseError:
        print 'Parse Error in the tag'
        return None
      


# Write to the JSON file.
def process_map(file_in, pretty = False):
    file_out = "{0}.json".format(file_in)
    data = []
    print 'in Process Map'
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data

# number of parking amenities
def make_pipeline1():
    pipeline = []
    pipeline = [
                 {"$match" : {"amenity" : "parking"}},
                 {"$group" : {"_id": "parking",
                              "count" : {"$sum" : 1 }}}
                ]
                 
               
    #print pipeline
    return pipeline
    
    
# number of waterway places
def make_pipeline2():
    pipeline = []
    pipeline = [
                {"$match" : {"pathtype.source" : "highway"}},
                 {"$group" : {"_id": "highway",
                              "count" : {"$sum" : 1 }}}
                ]
                 
               
    #print pipeline
    return pipeline
    

#places of worship
def make_pipeline3():
    pipeline = []
    pipeline = [
                {"$match" : {"amenity" : {"$exists":1},
                             "amenity": "place_of_worship"}},
                 {"$group" : {"_id": "$religion",
                              "count" : {"$sum" : 1 }}},
                {"$sort": {"count" : -1}},
                {"$limit" : 5 }
                ]
                 
               
    return pipeline
    


def aggregate(db, pipeline):
    return [doc for doc in db.osmmap.aggregate(pipeline)]


def mongoTest(jsonDoc):
    client = MongoClient('localhost', 27017)
    print 'hello'
    db = client.test
    #clear the existing collection from mongodb.
    result = db.osmmap.delete_many({})
    print result.deleted_count
    
    #insert the JSON document to the mongodb. 
    db.osmmap.insert(jsonDoc)

    #Aggregation pipeline for the query
    #pipeline = make_pipeline1()
    pipeline = make_pipeline1()
    #pipeline = make_pipeline3()
    result = aggregate(db,pipeline)
    print result
   
    #relRes = db.osmmap.find({'religion':{'$exists':True}})
    #for document in relRes:
    #    print(document)
    #result = db.osmmap.delete_many({})
    #print result.deleted_count
    
    print 'Done'
    
    
if __name__ == "__main__":
    data = process_map('interpreter', True)    
    mongoTest(data)

    