import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
from sklearn import metrics


df = pd.read_csv('5c.csv', encoding='latin-1')


st.title("ĐÁNH GIÁ RỦI RO TÍN DỤNG BẰNG MÔ HÌNH 5C")

uploaded_file = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    df.to_csv("data.csv", index = False)

df['TC']=(df['TC1']+df['TC2']+df['TC3']+df['TC4']+df['TC5'])/5
df['NL']=(df['NL1']+df['NL2']+df['NL3']+df['NL4'])/4
df['DK']=(df['DK1']+df['DK2']+df['DK3']+df['DK4']+df['DK5'])/5
df['V']=(df['V1']+df['V2']+df['V3']+df['V4']+df['V5']+df['V6'])/6
df['TS']=(df['TS1']+df['TS2']+df['TS3']+df['TS4'])/4


X = df[['TC','NL','DK','V','TS']]
y = df['PD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 12)

model = LogisticRegression()

model.fit(X_train, y_train)

yhat_test = model.predict(X_test)


score_train=model.score(X_train, y_train)
score_test=model.score(X_test, y_test)


confusion_matrix = pd.crosstab(y_test, yhat_test, rownames=['Actual'], colnames=['Predicted'])




menu = ["Mục tiêu của mô hình", "Xây dựng mô hình", "Sử dụng mô hình để dự báo"]
choice = st.sidebar.selectbox('Danh mục tính năng', menu)

if choice == 'Mục tiêu của mô hình':    
    st.subheader("Mục tiêu của mô hình")
    st.write("""
    ###### Mô hình được xây dựng để dự báo rủi ro tín dụng của khách hàng theo mô hình 5C.
    """)  
    
    st.image("hinh1.jpeg")
    

elif choice == 'Xây dựng mô hình':
    st.subheader("Xây dựng mô hình")
    st.write("##### 1. Hiển thị dữ liệu")
    st.dataframe(df.head(3))
    st.dataframe(df.tail(3))  
    
    st.write("##### 2. Build model...")
    
    st.write("##### 3. Evaluation")
    st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
    fig2=sns.heatmap(confusion_matrix, annot=True)
    st.pyplot(fig2.figure)
    he_so=model.coef_
    st.code("he so trong mo hinh: " + str(he_so))


    

    
elif choice == 'Sử dụng mô hình để dự báo':
    st.subheader("Sử dụng mô hình để dự báo")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1)
            st.dataframe(lines)
            # st.write(lines.columns)
            flag = True       
    if type=="Input":        
        TC1 = st.number_input('Sự sẵn sàng trả nợ của khách hàng')
        TC2 = st.number_input('Mối quan hệ trước đây của ngân hàng với khách hàng')
        TC3 = st.number_input('Có những ngân hàng khác đã thực hiện kinh doanh với những người đi vay')
        TC4 = st.number_input('Thông tin thu được từ CIC')
        TC5 = st.number_input('Ý kiến được tìm kiếm từ các ngân hàng khác về khách hàng')
        NL1 = st.number_input('Khách hàng có khả năng tạo ra tiền từ hoạt động kinh doanh phụ khi gặp khó khăn')
        NL2 = st.number_input('Khả năng kinh doanh để tạo ra đủ dòng tiền từ hoạt động kinh doanh chính')
        NL3 = st.number_input('Rủi ro dòng tiền không đạt kỳ vọng')
        NL4 = st.number_input('Số dư tiền mặt của khách hàng trong Tài khoản ngân hàng')
        DK1 = st.number_input('Các chính sách của Nhà nước có ảnh hưởng tốt đến khách hàng')
        DK2 = st.number_input('Chu kỳ của nền kinh tế hỗ trợ hoạt động kinh doanh của doanh nghiệp')
        DK3 = st.number_input('Sở thích tiêu dùng của khách hàng phù hợp với sản phẩm dịch vụ của khách hàng')
        DK4 = st.number_input('Yếu tố công nghệ có ảnh hưởng tích cực đến hoạt động kinh doanh của doanh nghiệp')
        DK5 = st.number_input('Hoạt động kinh doanh của khách hàng không ảnh hưởng đến môi trường')
        V1 = st.number_input('Các khoản đầu tư mà khách hàng đang thực hiện là hơp lý?')
        V2 = st.number_input('Khách hàng có khả năng huy động các nguồn vốn khi có nhu cầu?')
        V3 = st.number_input('Các khoản đầu tư mà khách hàng đang thực hiện là hiệu quả?')
        V4 = st.number_input('Số vốn đầu tư của chủ doanh nghiệp trong doanh nghiệp')
        V5 = st.number_input('Số tiền tài trợ so với vốn đối ứng của khách hàng')
        V6 = st.number_input('Khách hàng có còn nguồn vốn khác không?')
        TS1 = st.number_input('Tính đủ của tài sản thế chấp được đề xuất')
        TS2 = st.number_input('Giá trị tài sản đảm bảo đáp ứng yêu cầu')
        TS3 = st.number_input('Sự sẵn có của thị trường thứ cấp cho tài sản thế chấp')
        TS4 = st.number_input('Loại tài sản đảm bảo là hàng tồn kho, chứng từ hay tài sản cố định')

        TC=(TC1+TC2+TC3+TC4+TC5)/5
        NL=(NL1+NL2+NL3+NL4)/4
        DK=(DK1+DK2+DK3+DK4+DK5)/5
        V=(V1+V2+V3+V4+V5+V6)/6
        TS=(TS1+TS2+TS3+TS4)/4


        
        lines={'TC':[TC],'NL':[NL],'DK':[DK],'V':[V],'TS':[TS]}
        lines=pd.DataFrame(lines)
        st.dataframe(lines)
        flag = True
    
    if flag:
        st.write("Content:")
        if len(lines)>0:
            st.code(lines)
            X_1 = lines   
            y_pred_new = model.predict(X_1)
            pd=model.predict_proba(X_1)
            st.code("giá trị dự báo: " + str(y_pred_new))
            st.code("xác suất vỡ nợ của hộ là: " + str(pd))
