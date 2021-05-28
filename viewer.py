import pandas as pd
import numpy as np
import streamlit as st
import plotly as pt
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
plotly.offline.init_notebook_mode(connected=True)
import plotly.offline as py


import time
import os
from os.path import join
# Web Settings
st.set_page_config(
    page_title="BASF Sales Analysis by Ellin",
    layout="wide"
)




# upload

DOWNLOAD_PATH = './'

def upload_file():
    uploaded_file = st.file_uploader('xlsx 파일을 업로드 해주세요.')
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,
                        "FileType":uploaded_file.type,
                        "FileSize":uploaded_file.size}
        st.write(file_details)
        if file_details['FileName'].split('.')[-1] != 'xlsx':
            st.error('[ERROR] 파일 포맷이 엑셀(.xlsx) 가 아닙니다. 다시 업로드 해주세요.')
            return
        file_details['FileName'] = 'data.xlsx'
        with open(join(DOWNLOAD_PATH, file_details['FileName']), 'wb') as f:
            f.write(uploaded_file.getvalue())
            write_path = join(os.path.basename(DOWNLOAD_PATH), file_details['FileName'])
            st.success(f'{write_path} is saved!')

if not os.path.exists('data.xlsx'):
    upload_file()










# Logo
col1, col2, col3 = st.beta_columns([1,6,1])
with col1:
    st.write("")
with col2:
    st.write("")
with col3:
    st.image("./basf_blue.png")




########          ########          ########          ########          ########          ########          ######## 

def covid_impact_graph() :
    def calc_quantity_sales (sheet_name):
        df_data = pd.read_excel('data.xlsx',sheet_name = sheet_name, header = 1)
        df_data = df_data.dropna(axis=1)
        first_yr = df_data.iloc[:12, [1]].sum()
        first_yr = first_yr.iloc[0]
        second_yr = df_data.iloc[12:24, [1]].sum()
        second_yr = second_yr.iloc[0]
        third_yr = df_data.iloc[24:36, [1]].sum()
        third_yr = third_yr.iloc[0]
        fourth_yr = df_data.iloc[36:48, [1]].sum()
        fourth_yr = fourth_yr.iloc[0]
        fifth_yr = df_data.iloc[48:60, [1]].sum()
        fifth_yr = fifth_yr.iloc[0]
        Y= [first_yr,second_yr,third_yr,fourth_yr,fifth_yr]
        return Y
    
    list_yearly_quantity_sales = []

    for i in ['A','B','C','D','E','F']:
        list_yearly_quantity_sales.append(calc_quantity_sales(i))

    np_total = np.array(list_yearly_quantity_sales)

    year = list(range(2016,2021))               # X
    Quantity = np_total.sum(axis = 0)           # Y


    def calc_net_sales (sheet_name):
        df_data = pd.read_excel('data.xlsx',sheet_name = sheet_name, header = 1)
        df_data = df_data.dropna(axis=1)
        first_yr = df_data.iloc[:12, [2]].sum()
        first_yr = first_yr.iloc[0]
        second_yr = df_data.iloc[12:24, [2]].sum()
        second_yr = second_yr.iloc[0]
        third_yr = df_data.iloc[24:36, [2]].sum()
        third_yr = third_yr.iloc[0]
        fourth_yr = df_data.iloc[36:48, [2]].sum()
        fourth_yr = fourth_yr.iloc[0]
        fifth_yr = df_data.iloc[48:60, [2]].sum()
        fifth_yr = fifth_yr.iloc[0]
        Y= [first_yr,second_yr,third_yr,fourth_yr,fifth_yr]
        return Y

    list_yearly_net_sales = []

    for i in ['A','B','C','D','E','F']:
        list_yearly_net_sales.append(calc_net_sales(i))

    np_total = np.array(list_yearly_net_sales)
    Net_Sales = np_total.sum(axis = 0)           # Y




    # Plotting
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Original 2016-2020
    fig.add_trace(go.Scatter(name='Actual Net Sales ', x=year, y=Net_Sales), secondary_y=True)
    fig.add_trace(go.Bar(name='Actual Quantity Sales ', x=year, y=Quantity,width=[0.3, 0.3, 0.3, 0.3, 0.3]))

    # Predicted 2020 w/o COVID
    exp_quantity= np.array(708.2226-Quantity[4])
    yr_2020=np.array(2020)
    

    Net_Sales[4] = np.array(21343.7631)
    new_x = year[3:5]
    new_y = Net_Sales[3:5]
    
    fig.add_trace(go.Scatter(name='Expected Net Sales w/o COVID', x=new_x, y=new_y, mode='lines', line={'dash': 'dash', 'color': 'blue'}), secondary_y=True)
    fig.add_trace(go.Bar(name='Expected Quantity Sales w/o COVID', x=yr_2020, y=exp_quantity, width=[0.3]))


         
    # Add figure title
    fig.update_layout(
        width=1100,
        height=600,
        title_text="2016~2020yr Yearly Breakdown of UV Filter Sales vs. Month",
        yaxis_tickformat='M'
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    # Set x-axis title
    fig.update_xaxes(title_text='<b>Year</b>')
    fig.update_layout(
        barmode='stack',
        xaxis = dict(
            tickmode = 'array',
            tickvals = [2016,2017,2018,2019,2020],
        )
    )
    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Quantity Sales</b> (tons) ", range=[0,900], secondary_y=False)
    fig.update_yaxes(title_text="<b>Net Sales</b> (k€) ", secondary_y=True)

 
    st.write(fig)  
    st.markdown("""The graph clearly shows the COVID-19 impact on the UV Filter business.<br>
    <br>
    <br>
    <br>""", unsafe_allow_html = True)







########          ########          ########          ########          ########          ########          ########          

def total_relationship():
    Total = np.zeros(60)
    Total_Net_Sales = np.zeros(60)

    Month = ['January,2016',  'May, 2016', 'September,2016',
    'January,2017', 'May, 2017', 'September,2017',
    'January,2018', 'May, 2018', 'September,2018',
    'January,2019', 'May, 2019', 'September,2019',
    'January,2020', 'May, 2020', 'September,2020','December,2020'
    ]

    for i in ['A','B','C','D','E','F']:
        df_data = pd.read_excel('data.xlsx',sheet_name = i, header = 1)
        Total = Total + df_data['Total']
        Total_Net_Sales = Total_Net_Sales + df_data['Sales']


    fig = px.line(
                x=df_data.index, 
                y=[Total,Total_Net_Sales]
                )
    # Set Traces
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # fig.add_trace(go.Scatter(name='Net Profit Sales ', x=df_data.index, y=Total_Net_Sales), secondary_y=True)
    # fig.add_trace(go.Bar(name='Quantity Sales ', x=df_data.index, y=Total))
    
    fig.add_trace(go.Bar(name='Quantity Sales ', x=df_data.index, y=Total,marker=dict(color='red')))
    fig.add_trace(go.Scatter(name='Net Profit Sales ', x=df_data.index, y=Total_Net_Sales,marker=dict(color='royalblue')), secondary_y=True)
    
         
    # Add figure title
    fig.update_layout(
        width=1100,
        height=600,
        title_text="2016~2020yr Total UV Filter Sales vs. Month",
        yaxis_tickformat='M'
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    # Set x-axis title
    fig.update_xaxes(title_text='<b>Month, Year</b>')
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,59,60],
            ticktext = Month
        )
    )

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Quantity Sales</b> (tons) ",  secondary_y=False)
    fig.update_yaxes(title_text="<b>Net Profit Sales</b> (k€) ", secondary_y=True)

 
    st.write(fig)  
    




########          ########          ########          ########          ########          ########          ########          ########          






def overall_trend_quantity_by_month():

    Month = ['January,2016',  'May, 2016', 'September,2016',
    'January,2017', 'May, 2017', 'September,2017',
    'January,2018', 'May, 2018', 'September,2018',
    'January,2019', 'May, 2019', 'September,2019','December,2019'
    ]


    empty = []
    for i in ['A','B','C','D','E','F']:
            df_data = pd.read_excel('~/Desktop/Coeff_File.xlsx',sheet_name = i, header = 1)
            df_data = df_data.dropna(axis=1)

            month= df_data.index               # Output : RangeIndex (start=0,stop=60,step=1)
            x_input = [[x,1] for x in month]   # Output : [[0,1],[1,1]..]

            Y= df_data['Total'].values
            empty.append(Y)
    A,B,C,D,E,F = empty

    fig = px.line(
            x=month, 
            y=[A,B,C,D,E,F]
            )
    # Set Traces
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(name='A', x=month, y=A,showlegend=False))
    fig.add_trace(go.Scatter(name='B', x=month, y=B,showlegend=False))
    fig.add_trace(go.Scatter(name='C', x=month, y=C,showlegend=False))
    fig.add_trace(go.Scatter(name='D', x=month, y=D,showlegend=False))
    fig.add_trace(go.Scatter(name='E', x=month, y=E,showlegend=False))
    fig.add_trace(go.Scatter(name='F', x=month, y=F,showlegend=False))

    fig.update_layout(
        title_text='Monthly Sales Quantity by Each product vs. Year'
        )

    # Set x-axis title
    fig.update_xaxes(title_text="<b>Month, Year</b>")
    fig.update_yaxes(title_text="<b>Yearly sales qunatity</b> (Metric Tons)")
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [0,4,8,12,16,20,24,28,32,36,40,44,47,48],
            ticktext = Month
        ))
    st.write(fig) 
    



########          ########          ########          ########          ########          ########          ########          ########          




def overall_trend_net_sales_by_month():

    Month = ['January,2016',  'May, 2016', 'September,2016',
    'January,2017', 'May, 2017', 'September,2017',
    'January,2018', 'May, 2018', 'September,2018',
    'January,2019', 'May, 2019', 'September,2019','December,2019'
    ]


    empty = []
    for i in ['A','B','C','D','E','F']:
            df_data = pd.read_excel('~/Desktop/Coeff_File.xlsx',sheet_name = i, header = 1)
            df_data = df_data.dropna(axis=1)

            month= df_data.index               # Output : RangeIndex (start=0,stop=60,step=1)
            x_input = [[x,1] for x in month]   # Output : [[0,1],[1,1]..]

            Y= df_data['Sales'].values
            empty.append(Y)
    A,B,C,D,E,F = empty

    fig = px.line(
            x=month, 
            y=[A,B,C,D,E,F]
            )
    # Set Traces
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(name='A', x=month, y=A))
    fig.add_trace(go.Scatter(name='B', x=month, y=B))
    fig.add_trace(go.Scatter(name='C', x=month, y=C))
    fig.add_trace(go.Scatter(name='D', x=month, y=D))
    fig.add_trace(go.Scatter(name='E', x=month, y=E))
    fig.add_trace(go.Scatter(name='F', x=month, y=F))

    fig.update_layout(
        title_text='Monthly Net Sales by Each product vs. Month'
        )

    # Set x-axis title
    fig.update_xaxes(title_text="<b>Month, Year</b>")
    fig.update_yaxes(title_text="<b>Yearly Net Sales</b> (k€)")
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [0,4,8,12,16,20,24,28,32,36,40,44,47,48],
            ticktext = Month
        ))
    st.write(fig) 




########          ########          ########          ########          ########          ########          ########          ########    





def overall_trend_quantity_by_year():

    def calc_sales (sheet_name):
        df_data = pd.read_excel('~/Desktop/Coeff_File.xlsx',sheet_name = sheet_name, header = 1)
        df_data = df_data.dropna(axis=1)
        first_yr = df_data.iloc[:12, [1]].sum()
        first_yr = first_yr.iloc[0]
        second_yr = df_data.iloc[12:24, [1]].sum()
        second_yr = second_yr.iloc[0]
        third_yr = df_data.iloc[24:36, [1]].sum()
        third_yr = third_yr.iloc[0]
        fourth_yr = df_data.iloc[36:48, [1]].sum()
        fourth_yr = fourth_yr.iloc[0]
        Y= [first_yr,second_yr,third_yr,fourth_yr]
        return Y

    year=list(range(2016,2020))
    list_yearly_sales = []
    sheet_name = ['A','B','C','D','E','F']

    for i in sheet_name:
        list_yearly_sales.append(calc_sales(i))
    A,B,C,D,E,F = list_yearly_sales
    
    fig = px.line(
            x=year, 
            y=[A,B,C,D,E,F]
            )
    # Set Traces
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(name='A', x=year, y=A,showlegend=False))
    fig.add_trace(go.Scatter(name='B', x=year, y=B,showlegend=False))
    fig.add_trace(go.Scatter(name='C', x=year, y=C,showlegend=False))
    fig.add_trace(go.Scatter(name='D', x=year, y=D,showlegend=False))
    fig.add_trace(go.Scatter(name='E', x=year, y=E,showlegend=False))
    fig.add_trace(go.Scatter(name='F', x=year, y=F,showlegend=False))

    fig.update_layout(
        title_text='Yearly Sales Quantity by Each product vs. Year'
        )

    # Set x-axis title
    fig.update_xaxes(title_text="<b>Year</b>")
    fig.update_yaxes(title_text="<b>Yearly sales qunatity</b> (Metric Tons)")
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [2016,2017,2018,2019],
            ticktext = [2016,2017,2018,2019])
        )
    
    st.write(fig) 
    




########          ########          ########          ########          ########          ########          ########          ########    






def overall_trend_money_by_year():

    def calc_sales (sheet_name):
        df_data = pd.read_excel('~/Desktop/Coeff_File.xlsx',sheet_name = sheet_name, header = 1)
        df_data = df_data.dropna(axis=1)
        first_yr = df_data.iloc[:12, [2]].sum()
        first_yr = first_yr.iloc[0]
        second_yr = df_data.iloc[12:24, [2]].sum()
        second_yr = second_yr.iloc[0]
        third_yr = df_data.iloc[24:36, [2]].sum()
        third_yr = third_yr.iloc[0]
        fourth_yr = df_data.iloc[36:48, [2]].sum()
        fourth_yr = fourth_yr.iloc[0]
        Y= [first_yr,second_yr,third_yr,fourth_yr]
        return Y

    year=list(range(2016,2020))
    list_yearly_sales = []
    sheet_name = ['A','B','C','D','E','F']

    for i in sheet_name:
        list_yearly_sales.append(calc_sales(i))
    A=list_yearly_sales[0]; B=list_yearly_sales[1]; C=list_yearly_sales[2]
    D=list_yearly_sales[3]; E=list_yearly_sales[4]; F=list_yearly_sales[5]
    
    fig = px.line(
            x=year, 
            y=[A,B,C,D,E,F]
            )
    # Set Traces
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(name='A', x=year, y=A))
    fig.add_trace(go.Scatter(name='B', x=year, y=B))
    fig.add_trace(go.Scatter(name='C', x=year, y=C))
    fig.add_trace(go.Scatter(name='D', x=year, y=D))
    fig.add_trace(go.Scatter(name='E', x=year, y=E))
    fig.add_trace(go.Scatter(name='F', x=year, y=F))

    fig.update_layout(
        title_text='Yearly Net Sales by Each product vs. Year'
        )

    # Set x-axis title
    fig.update_xaxes(title_text="<b>Year</b>")
    fig.update_yaxes(title_text="<b>Yearly Net Sales</b> (k€)")
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [2016,2017,2018,2019],
            ticktext = [2016,2017,2018,2019])
        )
    
    st.write(fig) 





########          ########          ########          ########          ########          ########          ########          ########    





def each_prod_monthly_review():
    dict_description = {
        'A':'''- this is description<br>
        blue<br>
        blah''', # - exp
        'B':'# this is description', # title 1 (bolded)
        'C':'** this is description **', # just bolded exp
        'D':'## this is description',  # title 2 (not bolded)
        'E':'_ this is description _', # italicized
        'F':'this is description'} # just text

    Month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November','December']
    Year = ['2016','2017','2018','2019']
    for i in ['A','B','C','D','E','F']:
        df_data = pd.read_excel('~/Desktop/Coeff_File.xlsx',sheet_name = i , header = 1)
        df_data = df_data.dropna(axis=1)
        for month_year in df_data.index:
            df_data.loc[month_year, 'Year' ] = Year[month_year // 12 ]
            df_data.loc[month_year, 'Month'] = Month[month_year % 12]
        df = df_data[['Month','Year','Total','Ave. Unit Price','Sales']]  # use df.map to merge two columns (month+year)
        x = np.array(Month)
        Y = df['Total'].values
        Y = np.reshape(Y, (-1, 12))
        y1,y2,y3,y4 = Y

        table_trace1 = go.Table(
            domain=dict(x=[0, 0.4],
                        y=[0, 1]),
            columnwidth = [50] + [45, 55, 80,55],  # Table Column Width
            columnorder=[0, 1, 2, 3, 4],
            header = dict(height = 25,
                        values = [['<b>Month</b>'],['<b>Year</b>'], ['<b>Total</b>'],
                                    ['<b>Ave. Sales Price</b>'],['<b>Sales</b>']],
                        line = dict(color='rgb(50, 50, 50)'),
                        align = ['left'] * 5,
                        font = dict(color=['rgb(45, 45, 45)'] * 5, size=14),
                        fill = dict(color='#d562be')),
            cells = dict(values = [df['Month'].values, df['Year'].values, df['Total'].values.round(3), df['Ave. Unit Price'].values, df['Sales'].values],
                        line = dict(color='#506784'),
                        align = ['left'] * 5,
                        font = dict(color=['rgb(40, 40, 40)'] * 5, size=12),
                        format = [None] + [", .2f"] * 2 + [',.4f'],
                        suffix=[None] * 4,
                        height = 27,
                        fill = dict(color=['rgb(235, 193, 238)', 'rgba(228, 222, 249, 0.65)']))
        )

        trace1=go.Scatter(
            name='2016',
            x=x,
            y=y1,
            xaxis='x',
            yaxis='y',
            mode='lines',
            line=dict(width=2, color='royalblue')
        )

        trace2=go.Scatter(
            name='2017',
            x=x,
            y=y2,
            xaxis='x',
            yaxis='y',
            mode='lines',
            line=dict(width=2, color='indigo')
        )

        trace3=go.Scatter(
            name='2018',
            x=x,
            y=y3,
            xaxis='x',
            yaxis='y',
            mode='lines',
            line=dict(width=2, color='#b04553')
        )

        trace4=go.Scatter(
            name='2019',
            x=x,
            y=y4,
            xaxis='x',
            yaxis='y',
            mode='lines',
            line=dict(width=2, color='#af7bbd')
        )

        axis=dict(
            showline=True,
            zeroline=False,
            showgrid=True,
            mirror=True,
            ticklen=4,
            gridcolor='#ffffff',
            tickfont=dict(size=10)
        )

        layout1 = dict(
            width=1300,
            height=550,
            autosize=False,
            title=f'Product {i} : Sales Data',
            margin = dict(t=100),
            

            xaxis1=dict(axis, **dict(domain=[0.45, 1.0], anchor='y')),

            yaxis1=dict(axis, **dict(domain=[0, 1.0], anchor='x', hoverformat='.2f')),

            plot_bgcolor='rgba(228, 222, 249, 0.65)'
        )


        fig_sample = dict(data=[table_trace1, trace1,trace2,trace3,trace4], layout=layout1)
        st.write(fig_sample)
        st.markdown(dict_description[i], unsafe_allow_html = True)
        
    st.text('this is explaination')
    



########          ########          ########          ########          ########          ########          ########          ########  




def W_Sales_Overview():

    table_trace1 = go.Table(
            domain=dict(x=[0, 1],
                        y=[0, 1]),
            columnwidth = [120] + [130,130,130,130,130,130],  # Table Column Width
            columnorder=[0, 1, 2, 3, 4, 5, 6],
            header = dict(height = 25,
                        values = [['<b>W Coefficient</b>'],['<b>Product A</b>'], ['<b>Product B</b>'],
                                    ['<b>Product C</b>'],['<b>Product D</b>'],['<b>Product E</b>'],['<b>Product F</b>']],
                        line = dict(color='rgb(50, 50, 50)'),
                        align = ['left'] * 5,
                        font = dict(color=['rgb(45, 45, 45)'] * 5, size=14),
                        fill = dict(color='#d562be')),
            cells = dict(values = [['W_Net_Sales','W_cCM1','W_Net_Sales/W_cCM1'], [330.56597856,263.05948646,1.25662063], [34.00386133, 20.66198699,1.64572078],  [49.14664912,34.53511425,1.42309212],
                                                          [273.92240745,152.65663528,1.79436948], [178.46847626,94.06174324,1.89735455],  [328.54229607,278.15759691,1.18113724]  
                                    ],
                        line = dict(color='#506784'),
                        align = ['left'] * 5,
                        font = dict(color=['rgb(40, 40, 40)'] * 5, size=12),
                        format = [None] + [", .2f"] * 2 + [',.4f'],
                        suffix=[None] * 4,
                        height = 27,
                        fill = dict(color=['rgb(235, 193, 238)', 'rgba(228, 222, 249, 0.65)']))
        )
    layout1 = dict(
            width=1300,
            height=550,
            autosize=False,
            title='W Coefficient Analysis',
            margin = dict(t=100),
            

            plot_bgcolor='rgba(228, 222, 249, 0.65)'
        )
    w_figure = dict(data=[table_trace1], layout=layout1)
    st.write(w_figure)





########          ########          ########          ########          ########          ########          ########          ########  





def W_Sales_Yearly():
    def y_assign(sheet_name):
    
        df_data = pd.read_excel('data.xlsx',sheet_name = sheet_name, header = 1)
        df_data = df_data.dropna(axis=1)
        Y = df_data['Sales'].values
        return Y

    list_total = []

    for i in ['A','B','C','D','E','F']:
        list_total.append(y_assign(i))
        
    np_total = np.array(list_total)
    Y_NS = np_total.sum(axis = 0)
    Y_NS_COVID = Y_NS[:60]

    def call_out(sheet_name): 
        df_data = pd.read_excel('data.xlsx',sheet_name = sheet_name, header = 1)
        df_data = df_data.dropna(axis=1)
        Y = df_data['Total'].values
        return Y

    list_total = []
    unit_quantity = []

    for i in ['A','B','C','D','E','F']:
        list_total.append(call_out(i))
        x = list_total
        x = np.array(x)


        
    def normalization (product):
        x = np.array(product)
        norm_x = (x-x.mean())/x.std()
        return norm_x


    def x_assign (group_name):
        unit_quantity = np.array(group_name)
        unit_quantity = unit_quantity.reshape(6,12)

        x_input=[]

        for i in range (12):
            xvalue = unit_quantity[:,i].tolist()+[1]
            x_input.append(xvalue)


        x_input = np.array(x_input)
        x_input = np.reshape(x_input,(12,7))
        return x_input

    year_count = [2016,2017,2018,2019,2020]
    W_year = []

    for year in year_count :
        year_index = year-2016
        
        group_one =[]
        
        for i in [0,1,2,3,4,5]:
            a = x[i,12*year_index:12*(year_index+1)]
            group_one.append(normalization(a))
        X = x_assign(group_one)
        y = Y_NS_COVID[12*year_index:12*(year_index+1)].T
        W = y @ X @ np.linalg.inv(X.T @ X)
        W_year.append(W)

    W_year = np.array(W_year)
    W_year = W_year.T
    year_count = np.array(year_count)
    year_count = year_count.T

    table_trace1 = go.Table(
            domain=dict(x=[0, 1],
                        y=[0, 1]),
            columnwidth = [120] + [130,130,130,130,130,130],  # Table Column Width
            columnorder=[0, 1, 2, 3, 4, 5, 6],
            header = dict(height = 25,
                        values = [['<b>W Coefficient</b>'],['<b>Product A</b>'], ['<b>Product B</b>'],
                                    ['<b>Product C</b>'],['<b>Product D</b>'],['<b>Product E</b>'],['<b>Product F</b>']],
                        line = dict(color='rgb(50, 50, 50)'),
                        align = ['left'] * 5,
                        font = dict(color=['rgb(45, 45, 45)'] * 5, size=14),
                        fill = dict(color='#d562be')),
            cells = dict(values = [year_count, W_year[0].round(3),W_year[1].round(3),W_year[2].round(3),W_year[3].round(3),W_year[4].round(3),W_year[5].round(3)
                                    ],
                        line = dict(color='#506784'),
                        align = ['left'] * 5,
                        font = dict(color=['rgb(40, 40, 40)'] * 5, size=12),
                        format = [None] + [", .2f"] * 2 + [',.4f'],
                        suffix=[None] * 4,
                        height = 27,
                        fill = dict(color=['rgb(235, 193, 238)', 'rgba(228, 222, 249, 0.65)']))
        )
    layout1 = dict(
            width=1300,
            height=550,
            autosize=False,
            title='W Coefficient Analysis',
            margin = dict(t=100),
            

            plot_bgcolor='rgba(228, 222, 249, 0.65)'
        )
    w_figure = dict(data=[table_trace1], layout=layout1)
    st.write(w_figure)




########          ########          ########          ########          ########          ########          ########          ########  




def prediction():
    def calc_quantity_sales (sheet_name):
        df_data = pd.read_excel('data.xlsx',sheet_name = sheet_name, header = 1)
        df_data = df_data.dropna(axis=1)
        first_yr = df_data.iloc[:12, [1]].sum()
        first_yr = first_yr.iloc[0]
        second_yr = df_data.iloc[12:24, [1]].sum()
        second_yr = second_yr.iloc[0]
        third_yr = df_data.iloc[24:36, [1]].sum()
        third_yr = third_yr.iloc[0]
        fourth_yr = df_data.iloc[36:48, [1]].sum()
        fourth_yr = fourth_yr.iloc[0]
        fifth_yr = df_data.iloc[48:60, [1]].sum()
        fifth_yr = fifth_yr.iloc[0]
        Y= [first_yr,second_yr,third_yr,fourth_yr,fifth_yr]
        return Y
    
    list_yearly_quantity_sales = []

    for i in ['A','B','C','D','E','F']:
        list_yearly_quantity_sales.append(calc_quantity_sales(i))

    np_total = np.array(list_yearly_quantity_sales)

    year = list(range(2016,2022))               # X
    Quantity = np_total.sum(axis = 0)           # Y


    def calc_net_sales (sheet_name):
        df_data = pd.read_excel('data.xlsx',sheet_name = sheet_name, header = 1)
        df_data = df_data.dropna(axis=1)
        first_yr = df_data.iloc[:12, [2]].sum()
        first_yr = first_yr.iloc[0]
        second_yr = df_data.iloc[12:24, [2]].sum()
        second_yr = second_yr.iloc[0]
        third_yr = df_data.iloc[24:36, [2]].sum()
        third_yr = third_yr.iloc[0]
        fourth_yr = df_data.iloc[36:48, [2]].sum()
        fourth_yr = fourth_yr.iloc[0]
        fifth_yr = df_data.iloc[48:60, [2]].sum()
        fifth_yr = fifth_yr.iloc[0]
        Y= [first_yr,second_yr,third_yr,fourth_yr,fifth_yr]
        return Y

    list_yearly_net_sales = []

    for i in ['A','B','C','D','E','F']:
        list_yearly_net_sales.append(calc_net_sales(i))

    np_total = np.array(list_yearly_net_sales)
    Net_Sales = np_total.sum(axis = 0)           # Y




    # Plotting
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Original 2016-2020
    fig.add_trace(go.Scatter(name='Actual Net Sales ', x=year, y=Net_Sales), secondary_y=True)
    fig.add_trace(go.Bar(name='Actual Quantity Sales ', x=year, y=Quantity,width=[0.3, 0.3, 0.3, 0.3, 0.3]))

    # Predicted 2021 
    exp_quantity= np.array(575.38)
    yr_2021=np.array(2021)
    

    Net_Sales = np.append(Net_Sales,11235.69)
    new_x = year[4:6]
    new_y = Net_Sales[4:6]
    
    fig.add_trace(go.Scatter(name='Expected Net Sales in 2021', x=new_x, y=new_y, mode='lines', line={'dash': 'dash', 'color': 'blue'}), secondary_y=True)
    fig.add_trace(go.Bar(name='Expected Quantity Sales in 2021', x=yr_2021, y=exp_quantity, marker_color='rgb(255,50,33)', marker_line_color='rgb(8,48,107)',
                  opacity=0.6, width=[0.3]))


         
    # Add figure title
    fig.update_layout(
        width=1100,
        height=600,
        title_text="Forecast of the 2021yr UV Filter Sales vs. Year",
        yaxis_tickformat='M'
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    # Set x-axis title
    fig.update_xaxes(title_text='<b>Year</b>')
    fig.update_layout(
        barmode='stack',
        xaxis = dict(
            tickmode = 'array',
            tickvals = [2016,2017,2018,2019,2020,2021],
        )
    )
    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Quantity Sales</b> (tons) ", range=[0,900], secondary_y=False)
    fig.update_yaxes(title_text="<b>Net Sales</b> (k€) ", secondary_y=True)

 
    st.write(fig)  
    st.markdown("""The predicted quantity sales is 575.38 mT with a net sales of 11235.69 k€ .<br>
    <br>
    <br>
    <br>""", unsafe_allow_html = True)





 ########### ########### ########### ######### S T A R T ########### ########### ########### ###########         
st.title('BASF A-EMA/AR : UV Filter Sales Analysis')
st.subheader("Author : Ellin Hong")
st.markdown("""<br>
<br>
Due to COVID-19, the cosmetic business has been largely impacted. To quickly recover the negative effects on business, <br>
an accurate prediction of the future demand on UV filter products, which are the representative chemical products of BASF, is required.<br>
<br>""",unsafe_allow_html = True)


# Overview of Current
total_relationship()
st.markdown("""Based on the generated graphs, a proportional relationship between the total quantity sales and the net profit sales is expected.  <br> 
    To prove this hypothesis, further data analysis is conducted with the monthly breakdown of the quantity sales for each UV filter product.<br>
    <br>
    <br>""", unsafe_allow_html = True)


# COVID Impact
st.markdown("""A breakdown of yearly UV Filter sales is shown below with the predicted values for the quantity sales and net sales, assuming 
no COVID-19 impacts on business. For the calculation of prediction, a linear best-fit line is applied using data from 2016 through 2019;
<br>""", unsafe_allow_html = True)

st. write('''- Quantity Sales: ŷ = 16.64683X - 32918.36388
''',unsafe_allow_html = True)
st. write('''- Net Sales: ŷ = 1308.09585X - 2621009.85799
''',unsafe_allow_html = True)

covid_impact_graph()



# Intro to second graph
st.markdown("""<br>
<br>
To better observe the sales trend, a breakdown of the graph for each product is necessary. Thus, two graphs are genrated: one for sales quantity and the other for net sales. <br>
Since the data from year 2020 was impacted by COVID-19, it was assumed the year 2020 data should be not included for the scope of this analysis. <br>
The generated figures are shown as below: <br>
<br>""",unsafe_allow_html = True)




# Monthly Graphs
col1,col2 = st.beta_columns(2)
with col1:
    overall_trend_quantity_by_month()
with col2:
    overall_trend_net_sales_by_month()


st.markdown("""<br>
<br>
From the above figures, it is clear that Product D has been sold the most. In contrast to the monthly sales quantity graph, three products, Product A, D,and F, peaked higher
than the other three products on average. To clearly see the discrepancy between the trends, two figures are represented as below for a yearly sales quantity graph and a yearly net sales graph.<br>

<br>""",unsafe_allow_html = True)


# Yearly graphs
col1,col2 = st.beta_columns(2)          # Aligning two graphs on the same level
with col1:
    overall_trend_quantity_by_year()
with col2:
    overall_trend_money_by_year()





# Linear Regression
st.markdown("""<br>
<br>
 Linear Regerssion is employed to determine the products that contribute most to the net sales. Since there are 6 products, linear regression 
 is evaluated for 6 different effects or regression coefficients denoted as W.<br>

<br>""",unsafe_allow_html = True)
# Expander for Linear Regression Description
with st.beta_expander ("See Description for Linear Regression"):
    st.image("./linear_regression.png")
    st. write(""" 
    In statistics, linear regression is a linear approach to modelling the relationship between a scalar response and one or more 
    explanatory variables (also known as dependent and independent variables). <br>
    """,unsafe_allow_html = True)
    st.write("""
    Linear regression has many practical uses. Most applications fall into one of the following two broad categories: <br>
    """,unsafe_allow_html = True)
    st.write('''- If the goal is prediction, forecasting, or error reduction, linear regression can be used to fit a predictive model 
    to an observed data set of values of the response and explanatory variables. After developing such a model, if additional values of 
    the explanatory variables are collected without an accompanying response value, the fitted model can be used to make a prediction of
    the response.<br>''',unsafe_allow_html = True)
    st. write('''- If the goal is to explain variation in the response variable that can be attributed to variation in the explanatory variables, 
    linear regression analysis can be applied to quantify the strength of the relationship between the response and the explanatory 
    variables, and in particular to determine whether some explanatory variables may have no linear relationship with the response 
    at all, or to identify which subsets of explanatory variables may contain redundant information about the response.
       ''',unsafe_allow_html = True)

  




W_Sales_Overview()

st.markdown("""
ahahahahahha
<br>""",unsafe_allow_html = True)


W_Sales_Yearly()

st.markdown("""
ahahahahahha
<br>""",unsafe_allow_html = True)



# Each Product Graph
with st.beta_expander ("See Monthly Sales Graph of Products"):
    each_prod_monthly_review()



prediction()