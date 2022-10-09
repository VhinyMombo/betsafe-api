import pandas as pd
import numpy as np
import streamlit as st
from BestBet import BestBet
import matplotlib.pyplot as plt
from PIL import Image


im = Image.open("ic3.png")
st.set_page_config(
    layout="wide",
    page_icon=im#"🦈"
)
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.write("")
with col2:
    st.image("ic3.png", use_column_width = True)
with col3:
    st.write("")

prob_A = None
prob_B = None
prob_N = None

submit = 0

st.sidebar.markdown('## Follow all steps')
odd_A = st.sidebar.number_input('Enter odd for event A',0.0)
odd_N = st.sidebar.number_input('Enter odd for event N',0.0)
odd_B = st.sidebar.number_input('Enter odd for event B',0.0)

proba_bool = st.sidebar.radio(
    'Do you have probabilities associated for each event ? ',
    ['Yes', 'No'],
    index = 1)
if proba_bool == 'Yes':
    prob_A = st.sidebar.number_input('Probabilities A',0.0,1.0,step = 0.01)
    prob_N = st.sidebar.number_input('Probabilities N',0.0,1.0,step = 0.01)
    prob_B = st.sidebar.number_input('Probabilities B',0.0,1.0,step = 0.01)
    if np.sum([prob_B,prob_A, prob_N ]) != 1.0 and proba_bool == 'Yes':
        st.sidebar.warning( 'sum of probabilities should be 1')

    amount = st.sidebar.number_input('How much do you want to bet?',0.0)

    st.sidebar.text('''




    ''')
    if (all([amount!=0.00, odd_A !=0 , odd_B !=0,
    amount !=None, odd_A !=None, odd_N !=None, odd_B !=None]) and np.sum([prob_B,prob_A,prob_N ]) == 1.0):
        submit = st.sidebar.button('Submit')
else:
    amount = st.sidebar.number_input('How much do you want to bet?',0.0)

    if all([amount!=0.00, odd_A !=0, odd_N !=0, odd_B !=0,
    amount !=None, odd_A !=None, odd_B !=None, odd_N !=None]):
        submit = st.sidebar.button('Submit')

if submit:  
    st.subheader('Results for entries')
    submit = 0
    C = np.array([odd_A,odd_N, odd_B])
    if (np.array([prob_A, prob_N,prob_B]) == None).sum() == 3:
        model = BestBet(C)
    else:
        model = BestBet(C,Proba_bookmaker=np.array([prob_A, prob_N,prob_B]))

    model_defined = True

        
    st.markdown('#### Allocation based on Kelly Criterion')
    st.text('''
    That indicates what the portion of your capital to bet for the long term return'''
    )
    kelly =  model.kelly_criterion()
    st.dataframe(
        kelly
    )


    kelly_p = kelly.loc[:, (kelly != 0).any(axis=0)]

    montant= kelly_p * amount

    advice1 = "For the first strategy we advice you to bet : \n" 
    adv = ""
    for col in montant.columns:
        adv += "    * {} on the issue {} \n".format(round(montant[col][0],2),col)

    st.text(advice1+adv)

    model.strategy(amount)

    if model.check_risk_free()=='Arbitrage existe':
        st.markdown('#### Any Arbitrage')
        st.warning(model.check_risk_free())
    
    st.markdown('#### Dataframe simulating strategies paths')
    st.dataframe(model.phi)
    st.markdown('#### Graphic variance - return')
    st.plotly_chart(model.graph_VarianceReturn(),use_container_width=True)


    st.markdown('#### Best Bet based on variance minimization')
    best = model.best_bet(type='variance')
    st.dataframe(pd.DataFrame( 
        best['bet']
        ))

    A = str(best['bet']['A'].to_numpy()[0].round(2))
    N = str(best['bet']['N'].to_numpy()[0].round(2))
    B = str(best['bet']['B'].to_numpy()[0].round(2)) 
    
    advice = ''' 
        The safest best is  : 
            {} on the issue A, 
            {} on the issue B,
            {} on the issue N
        
        For a gain/loss of :
            {} for A,
            {} for B and
            {} for N.
            '''.format(
                A,
                B,
                N,
                best['Gain/Loss'][0][0].round(2),
                best['Gain/Loss'][0][1].round(2),
                best['Gain/Loss'][0][2].round(2)
            )

    st.text(advice)   
    if all([best['Gain/Loss'][0][0].round(2)<0,best['Gain/Loss'][0][2].round(2)<0,best['Gain/Loss'][0][1].round(2)<0]):
        st.warning(''' You'd better not bet on this Games, it's too risky!''')









