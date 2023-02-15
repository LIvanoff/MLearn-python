from ARC.apriori import Apriori

if __name__ == "__main__":
    arc = Apriori(filename='data.json', min_length=3)
    arc.apriori(excel=True)