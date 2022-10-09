import pandas as pd
import numpy as np
import streamlit as st
from BestBet import BestBet
import matplotlib.pyplot as plt
from PIL import Image


def page_2_issues():
    prob_A = None
    prob_B = None
    submit = 0

    st.sidebar.markdown('## Follow all steps')
    odd_A = st.sidebar.number_input('Enter odd for event A',0.0)
    odd_B = st.sidebar.number_input('Enter odd for event B',0.0)
    proba_bool = st.sidebar.radio(
        'Do you have probabilities associated for each event ? ',
        ['Yes', 'No'],
        index = 1)
    if proba_bool == 'Yes':
        prob_A = st.sidebar.number_input('Probabilities A',0.0,1.0,step = 0.01)
        prob_B = st.sidebar.number_input('Probabilities B',0.0,1.0,step = 0.01)
        if np.sum([prob_B,prob_A ]) != 1.0 and proba_bool == 'Yes':
            st.sidebar.warning( 'sum of probabilities should be 1')

        amount = st.sidebar.number_input('How much do you want to bet?',0)

        if (all([amount!=0.00, odd_A !=0 , odd_B !=0,
        amount !=None, odd_A !=None, odd_B !=None]) and np.sum([prob_B,prob_A ]) == 1.0):
            submit = st.sidebar.button('Submit')
    else:
        amount = st.sidebar.number_input('How much do you want to bet?',0.0)

        if all([amount!=0.00, odd_A !=0 , odd_B !=0,
        amount !=None, odd_A !=None, odd_B !=None]):
            submit = st.sidebar.button('Submit')

    col1, col2 = st.columns([3,1])
    col2.subheader('recents entries')

    if submit:  
        st.subheader('Results for entries')
        submit = 0
        C = np.array([
            odd_A, 
            odd_B
        ])
        C = np.array([odd_A, odd_B])
        model = BestBet(C)
        model_defined = True
        model.strategy(amount)

        col1.markdown('#### Any Arbitrage')
        if model.check_risk_free()=='Arbitrage existe':
            col1.warning(model.check_risk_free())
        col1.markdown('#### Dataframe simulating strategies paths')
        col1.dataframe(model.phi)
        col1.markdown('#### Graphic variance - return')
        col1.plotly_chart(model.graph_VarianceReturn(),use_container_width=True)


        col1.markdown('#### Best Bet based on variance minimization')
        best = model.best_bet(type='variance')
        col1.dataframe(pd.DataFrame( 
            best['bet']
            ))

        A = str(best['bet']['A'].to_numpy()[0].round(2))
        B = str(best['bet']['B'].to_numpy()[0].round(2)) 
        res = ' '.join([A, B])
        gl = ' ' + str(best['Gain/Loss'][0][0].round(2)) + ''' pour A contre 
            ''' + str(best['Gain/Loss'][0][1].round(2)) + ' pour B'

        advice = ''' 
        <p> 
            The safest best is  : <br>  ''' + A +\
            ''' on the issue A, 
            <br> ''' + B  +\
            ''' on the issue B and 
            <br> ''' +\
            '''For a gain/loss of ''' +\
            gl +\
        '''</p>''' 
        col1.markdown(advice,unsafe_allow_html=True)


        if all([best['Gain/Loss'][0][0].round(2)<0,best['Gain/Loss'][0][1].round(2)<0]):
            col1.warning(''' You'd better not bet on this Games, it's too risky!''')


def page_3_issues():
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

        amount = st.sidebar.number_input('How much do you want to bet?',0)

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


    col1, col2 = st.columns([3,1])
    col2.subheader('recents entries')

    if submit:  
        col1.subheader('Results for entries')
        submit = 0
        C = np.array([odd_A,odd_N, odd_B])
        model = BestBet(C)
        model_defined = True
        model.strategy(amount)

        col1.markdown('#### Any Arbitrage')
        if model.check_risk_free()=='Arbitrage existe':
            col1.warning(model.check_risk_free())
        col1.markdown('#### Dataframe simulating strategies paths')
        col1.dataframe(model.phi)
        col1.markdown('#### Graphic variance - return')
        col1.plotly_chart(model.graph_VarianceReturn(),use_container_width=True)


        col1.markdown('#### Best Bet based on variance minimization')
        best = model.best_bet(type='variance')
        col1.dataframe(pd.DataFrame( 
            best['bet']
            ))

        A = str(best['bet']['A'].to_numpy()[0].round(2))
        N = str(best['bet']['N'].to_numpy()[0].round(2))
        B = str(best['bet']['B'].to_numpy()[0].round(2)) 
        res = ' '.join([A, B])
        gl = ' ' + str(best['Gain/Loss'][0][0].round(2)) + ''' pour A contre 
            ''' + str(best['Gain/Loss'][0][1].round(2)) + ''' pour N
            ''' + str(best['Gain/Loss'][0][2].round(2)) + ' pour B'

        advice = ''' 
        <p> 
            The safest best is  : <br>  ''' + A +\
            ''' on the issue A, 
            <br> ''' + B  +\
            ''' on the issue B and 
            <br> ''' +\
            N + ''' on the issue N. <br>  For a gain/loss of ''' +\
            gl +\
        '''</p>''' 
        col1.markdown(advice,unsafe_allow_html=True)

        if all([best['Gain/Loss'][0][0].round(2)<0,best['Gain/Loss'][0][2].round(2)<0,best['Gain/Loss'][0][1].round(2)<0]):
            col1.warning(''' You'd better not bet on this Games, it's too risky!''')
