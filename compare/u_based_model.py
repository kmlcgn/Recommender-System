import numpy as np
import pandas as pd

class CarettaSVDUserBasedModel:
    
    def __init__(self, customerNo):
        self.customerNo =  customerNo
        print('CarettaSVD model is created')

    def top_cosine_similarity(self, data, ref_customer_id):

        ## Select user X and 100 concepts of it
        user_row = np.array(data[ref_customer_id, :]) 

        ## Apply cosine similarity.
        similarity = np.dot(data, user_row)

        ## np.argsort sorts users in ascending order, so it should be reversed and the first item which is the user itself should be excluded.
        sort_indeces = np.argsort(similarity)
        sort_indeces = sort_indeces[::-1]
        
        return sort_indeces[1:]
    
    def prepare_data(self, customerNo):
        sales = pd.read_csv('/Users/kmlcgn/Downloads/sales.csv')
        sales_original = sales.copy()
        
        sales_original = sales_original[sales_original.Quantity > 0]
        sales_original['CustomerNo'] = sales_original['CustomerNo'].fillna(0).astype('float')
        sales_original['CustomerNo'] = sales_original['CustomerNo'].astype('int')

        sales_original = sales_original.groupby(by=['CustomerNo','ProductNo','ProductName']).sum().reset_index()

        ## Drop the columns that won't be used.
        sales.drop(['TransactionNo', 'Date', 'ProductName', 'Country'], inplace=True, axis=1)
        
        ## Some of the sales have negative values, remove these sales.
        sales = sales[sales.Quantity > 0]

        ## Create TotalAmount column which represents the multipication of product's price and the quantity that user has bought.
        sales['TotalAmaount'] = sales['Price'] * sales['Quantity']

        ## CustomerNo column values are float type, convert them into integer values.
        sales['CustomerNo'] = sales['CustomerNo'].fillna(0).astype('float')
        sales['CustomerNo'] = sales['CustomerNo'].astype('int')

        ## Merge the same product purchases of all users.
        sales = sales.groupby(by=['CustomerNo','ProductNo']).sum().reset_index()
        
        sales.drop(['Price', 'Quantity'], inplace=True, axis=1)

        return sales, sales_original
     
    def create_ratings_matrix(self, sales):
        ## Create an exmpty matrix filled with zeros.
        ratings_matrix = np.zeros(shape=(len(sales.ProductNo.unique()), len(sales.CustomerNo.unique())), dtype = np.uint8)

        ## Append products, sales and total amounts in different arrays.
        products_array = sales.ProductNo.values
        customers_array = sales.CustomerNo.values
        amounts_array = sales.TotalAmaount.values

        ## Create dictionaries to map real product numbers to reference numbers. ie. 13490 : 1 and vice versa.
        unique_products_dict_ref_real = {}
        unique_products_dict_real_ref = {}

        unique_products = np.unique(products_array)

        ## Fill dictionaries with references and original values.
        for idx, res in enumerate(unique_products):
            unique_products_dict_ref_real[idx] = res
            unique_products_dict_real_ref[str(res)] = idx
        
        ## Do the same steps for customers array.
        unique_customers_dict_ref_real = {}
        unique_customers_dict_real_ref = {}

        unique_customers = np.unique(customers_array)

        for idx, res in enumerate(unique_customers):
            unique_customers_dict_ref_real[idx] = res
            unique_customers_dict_real_ref[res] = idx

        ## Iterate over customers and products array and fill the final matrix.
        for i in range(len(amounts_array)):
            u = customers_array[i]
            m = products_array[i]

            ref_to_u = unique_customers_dict_real_ref[u]
            ref_to_m = unique_products_dict_real_ref[m]

            ## Fill the empty matrix with values.
            ratings_matrix[ref_to_m, ref_to_u] = amounts_array[i]
            ## Products are represented by rows, users are represented by columns.

        return ratings_matrix, unique_customers_dict_real_ref, unique_customers_dict_ref_real

    def get_SVD(self, ratings_matrix):

        ## Create U, S, and V matrices from original matrix.
        U, S, V = np.linalg.svd(ratings_matrix)
        return U, S, V
    
    def get_recommendations(self, sales_original, k, unique_customers_dict_real_ref, unique_customers_dict_ref_real, U, S, V, desired):

        selected_customer_id = self.customerNo

        ref_customer_id = unique_customers_dict_real_ref[selected_customer_id]

        ## List products that selected user already bought.
        already_bought = (sales_original.loc[sales_original['CustomerNo'] == selected_customer_id].ProductName.values)

        ## Select first k elements of Users x Concepts matrix.
        sliced_matrix = np.array(V[:,:k])
     
        ## top_cosine_similarity method returns all indeces (all correlation between users, beginning with the most similar user.)
        all_indeces = self.top_cosine_similarity(sliced_matrix, ref_customer_id)

        ## Append real customer ids of other users in order.
        similar_user_list = []

        for ref in all_indeces:
            real_customer_id = unique_customers_dict_ref_real[ref]
            similar_user_list.append(real_customer_id)

        recommend_list = []

        ## i shows the i'th similar user.
        i = 0

        ## Until desired number of recommendations are made, iterate over the similar_user_list.
        while(len(recommend_list) < desired):
            ## List the next most similar user's purchases.
            next_similar_user_purchases = sales_original[sales_original.CustomerNo == similar_user_list[i]].ProductName.values
            
            ## Do not recommend products that are already bought by the selected user.
            common_products = np.intersect1d(next_similar_user_purchases, already_bought)

            for j in next_similar_user_purchases:
                if j not in common_products:
                    if j not in recommend_list:
                        recommend_list.append(j)
                
            i += 1
            
        return recommend_list[:desired]





