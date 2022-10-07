def define_model(a_odd, b_odd, submit):
    global model
    global model_defined
    global tbl

    tbl = False
    if n_clicks1 > 0:
        if a_odd is None or b_odd is None:  # \
            # and a_prob is None and b_prob is None and n_prob is None:
            # PreventUpdate prevents ALL outputs updating
            raise dash.exceptions.PreventUpdate
        C = np.array([a_odd, b_odd])
        model = BestBet(C)
        model_defined = True
        n_clicks1 = 0
