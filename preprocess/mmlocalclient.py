import re
from pymetamap import MetaMap
from time import sleep
import os

class METAMAP():
    def __init__(self) -> None:
        path = os.getcwd()
        
        # Setup UMLS Server
        metamap_base_dir = path + '/public_mm'
        metamap_bin_dir = '/bin/metamap18'
        metamap_pos_server_dir = '/bin/skrmedpostctl'
        metamap_wsd_server_dir = '/bin/wsdserverctl'

        # Start servers
        os.system(metamap_base_dir + metamap_pos_server_dir + ' start') # Part of speech tagger
        os.system(metamap_base_dir + metamap_wsd_server_dir + ' start') # Word sense disambiguation 

        # Sleep a bit to give time for these servers to start up
        sleep(10)

        self.mm = MetaMap.get_instance(metamap_base_dir + metamap_bin_dir)
        self.keys_of_interest = ['preferred_name', 'cui']

    def get_enti_from_mm(self, concepts, klist):
        entities = []
        for concept in concepts:
            if concept.__class__.__name__ == 'ConceptMMI':
                conc_dict = concept._asdict()
                conc_list = [conc_dict.get(kk) for kk in klist]
                key_value_pairs = zip(klist, conc_list)
                my_dict = {key: value for key, value in key_value_pairs}
                entities.append(my_dict)

        return entities
    
    def remove_non_ascii(self, input_string):
        # Use a regular expression to match non-ASCII characters and replace them with an empty string
        return re.sub(r'[^\x00-\x7F]+', '', input_string)
    
    def get_entities(self, query):
        query = [self.remove_non_ascii(query)]
        cons, errs = self.mm.extract_concepts(query,
                                    word_sense_disambiguation = True,
                                    strict_model=True,
                                    composite_phrase = 1,
                                    exclude_sts=['qlco', 'qnco', 'tmco', 'ftcn'],
                                    prune = 30)
        list_of_dicts = self.get_enti_from_mm(cons, self.keys_of_interest)

        # extracted_values = [d["preferred_name"].lower() for d in list_of_dicts]
        extracted_values = [d["cui"] for d in list_of_dicts]
        
        if extracted_values:
            return set(extracted_values)
        else:
            return None
        

if __name__ == '__main__':
    metamap = METAMAP()

    #do a sample test
    question = "Do preoperative statins reduce atrial fibrillation after coronary artery bypass grafting?"
    get_ = metamap.get_entities(question)
    print(get_)
