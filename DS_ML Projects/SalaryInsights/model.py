from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
#Import os package
import os
import pickle

def read_data():
    # set the path of the raw data
    raw_file_path = os.path.join(os.path.pardir, 'data', 'raw')
    raw_data_file_path = os.path.join(raw_file_path, 'salary_data_1.xlsx')
    raw_data = pd.read_excel(raw_data_file_path)
    
    return raw_data

def GetJobTitle(jobtitle):
    engineer = ["engineer", "analyst", "animator", "data scientist","technician", "engineering", "qa", "quality","tech", 
                 "control", "software","consultant", "designer", "developer", "specialist"]
    account = ["account", "claims"]
    hr  = ["talent", "hr", "recruiting", "recruiter", "human", "resources"]
    scientist = ["scientist", "research", "r&d", "technologist", "pathologist"]
    sales = ["sales", "marketing"]
    admin = ["administrator", "office", "admin"]
    media = ["reporter", "journalist", "media"]
    it  = ["support", "it"]
    medical = ["health", "medical", "mental", "physical","therapist", "clinical", "nursery", "nursing", "nurse", "psychologist"]
    teacher = ["teacher","lecturer", "tutor","instructor","trainer", "education", "training", "librarian", "educator",  
               "facilitator", "music", "learning", "library"]
    others = ["environmental", "environment","digital","communications","partner", "clerk", "coordinator", "visual","start-up",
               "store", "stripper", "shipping", "vet", "wedding", "brewer", "paralegal", "receptionist", "student","others",
               "cashier","intern","cook", "chef", "borrow", "entry", "adjunct", "supervisor", "superviso", "host", "artist", 
               "reception", "refurbisher", "operations", "productions", "production", "producer", "lab", "biller", "loan", 
               "line", "party", "coach", "homemaker", "goddess", "broker", "attendant", "firefighter", "executive", 
               "electrician", "assembler", "solutions", "customer", "care", "barista", "collector", "club", "ceo", "chief", 
               "case", "associate", "youth", "agent", "member", "owner", "operator", "property","ta","culinary", "server", 
               "busser", "independent", "contractor", "public", "retired", "bookseller","attorneu", "ea", "food", "maid", 
               "museum","keyholder", "cna", "experience" "package", "handler", "project",  "shift", "attorney",  "boss", 
               "interviewer", "lead", "bookkeeper", "farmer", "manager",  "senior", "driver", "paraprofessional",  "vp", 
               "flipper", "apprentice", "key", "phd", "receiver", "bakery", "concierge", "content", "worker", "delivery", 
               "freelance", "freelancer", "graduate", "employee", "head", "homebrew", "expert", "homemsker", "photographer", 
               "player", "tour", "guide", "resident", "companion", "dyer", "guest",  "head", "events", "temp",  "bid", "nanny",
               "bagger", "document", "teller", "advertising", "naturalist", "crew", "general", "transcriptionist", "in-charge",
               "front", "mason", "worship", "zookeeper", "level", "bartender", "arborist", "courier", "archaeologist", "HR", 
               "musician", "processor", "advisor", "acupuncturist", "housekeeper", "president",  "floor", "experience","csr", 
               "ranger", "esthetician", "registrar", "boxer", "director", "billing", "finance",  "landscaper", "craftsman", 
               "merchandiser",  "certified" ]
    
    title = jobtitle
        
    title = title.strip().lower()
    for i in range(len(engineer)):
        if (engineer[i] in title):
             title = "Engineer"
    for i in range(len(teacher)):
        if (teacher[i] in title):
             title = "Teacher"
    for i in range(len(account)):
        if (account[i] in title):
            title = "Accountant"
    for i in range(len(hr)):
        if (hr[i] in title):
             title = "HR"
    for i in range(len(scientist)):
        if (scientist[i] in title):
            title = "Scientist" 
    for i in range(len(sales)):
        if (sales[i] in title):
             title = "Sales"
    for i in range(len(admin)):
        if (admin[i] in title):
            title = "Admin"
    for i in range(len(media)):
        if (media[i] in title):
             title = "Media"
    for i in range(len(it)):
        if (it[i] in title):
            title = "IT"
    for i in range(len(medical)):
        if (medical[i] in title):
            title = "Medical"
    for i in range(len(others)):
        if (others[i] in title):
             title = "Others"
    
    return title


def  process_data(raw_data):
            raw_data['Job Title'] = raw_data['Generic Job Title'].map(lambda x : GetJobTitle(x))
            raw_data.drop(['Generic Job Title'], axis = 1, inplace=True)
            raw_data['Experience'] = round(raw_data['Experience'])
            raw_data['Gender Encoded'] = LabelEncoder().fit_transform(raw_data['Gender'])
            raw_data['Job Title Encoded'] = LabelEncoder().fit_transform(raw_data['Job Title'])
            raw_data = raw_data[['Experience', 'Gender', 'Gender Encoded', 'Job Title', 'Job Title Encoded', 'Salary']]
            raw_data.drop(['Gender'], axis = 1, inplace=True)
            raw_data.drop(['Job Title'], axis = 1, inplace=True)
            return raw_data
            
def write_data(processed_data):
            processed_data_path = os.path.join(os.path.pardir, 'data', 'processed')
            write_processed_data_path = os.path.join(processed_data_path, 'processed_data.xlsx')
            raw_data.to_excel(write_processed_data_path, index=False)
            processed_data = pd.read_excel(write_processed_data_path)
            return processed_data

def running_model(processed_data1):
            train, test = train_test_split(processed_data, test_size = 0.2)
            features = ['Experience', 'Gender Encoded', 'Job Title Encoded']
            X_train = train[features]
            y_train = train['Salary']
            X_test = test[features]
            y_test = test['Salary'] 
            # Creating the model
            clf = RandomForestRegressor(n_estimators=100, random_state=0)
            # Training the model
            clf = clf.fit(X_train, y_train)
            # Saving model to disk
            pickle.dump(clf, open('../model.pkl','wb'))
            # Loading model to compare the results
            model = pickle.load(open('../model.pkl','rb'))
            print(model.predict(X_test))


if __name__ == '__main__':
                  raw_data = read_data()
                  processed_data = process_data(raw_data)
                  processed_data1 = write_data(processed_data)
                  running_model(processed_data1)
