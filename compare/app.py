import streamlit as st
from p_based_model import *
from u_based_model import *

st.title('Caretta SVD Recommendation Engine')

st.markdown("""
*Recommend New Products Based on Similar Users to You.*
""")

for i in range(4):
    st.write("")

customerID = st.number_input("Enter Your Customer ID, ie. 18069, 19740", step= 1)
desired_n_of_recom = st.number_input("How many recommendations would you like to see per item? ie. 2, 5, 10", step= 1)

#--------------------------------------------------------------------
if (st.button("Recommend Products")):

    st.markdown("""*Product Based Recommendations Loading.*""")
    
    carettaSvd_P_Based = CarettaSVDProductBasedModel(customerID)

    (sales, sales_original, unique_products_user_bought) = carettaSvd_P_Based.prepare_data(customerID)

    (ratings_matrix, unique_products_dict_real_ref, unique_products_dict_ref_real) = carettaSvd_P_Based.create_ratings_matrix(
        sales)

    (U, S, V) = carettaSvd_P_Based.get_SVD(ratings_matrix)

    recommendations, len_of_recoms = carettaSvd_P_Based.get_recommendations(unique_products_user_bought, sales_original, 200, unique_products_dict_real_ref, unique_products_dict_ref_real, U, S, V, desired_n_of_recom)

    df = pd.DataFrame({"Product Based Recommendations": recommendations})

    st.dataframe(df, width=600, height=900)
    

#---------------------------------------------------------------------

    st.markdown("""*User Based Recommendations Loading.*""")
    
    carettaSvd_U_Based = CarettaSVDUserBasedModel(customerID)

    (sales, sales_original) = carettaSvd_U_Based.prepare_data(customerID)

    (ratings_matrix_ub, unique_customers_dict_real_ref_ub, unique_customers_dict_ref_real_ub) = carettaSvd_U_Based.create_ratings_matrix(sales)

    (U_ub, S_ub, V_ub) = carettaSvd_U_Based.get_SVD(ratings_matrix_ub)

    recommendations_ub = carettaSvd_U_Based.get_recommendations(sales_original, 200, unique_customers_dict_real_ref_ub, unique_customers_dict_ref_real_ub, U_ub, S_ub, V_ub, len_of_recoms)

    df = pd.DataFrame({"User Based Recommendations": recommendations_ub})

    st.dataframe(df, width=600, height=900)
    

#--------------------------------------------------------------------

    st.markdown("""*Similarity of Recommendations.*""")
    common_elements = list(set(recommendations).intersection(recommendations_ub))
    st.write(round(len(common_elements)/len(recommendations) * 100))


