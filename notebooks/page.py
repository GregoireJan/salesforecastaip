def properties():
    return {"pageTitle": "Page"}
 
 
def presenter():
    user = app.current_session.user
    app.page_variables.increase = 0
 
    app._private.page_widgets = []      # variable defined by the Design Studio Editor to store widgets
    #configuration of:  actionLink1
    app._private.page_widgets.append(json.loads('{"widgetId":"actionLink1","name":"actionLink","value":{"label":"Click for refresh",
    "textStyle":"ux- body1","textColor":"blue","eventType":"refresh"},"position":"#actionLink1"}'))
    #configuration of:  label1
    app._private.page_widgets.append(json.loads('{"widgetId":"output1","name":"output","value":{"label":"Simple text","class":"",
    "textStyle":"ux-body1","textColor":"grey-p3"},"position":"#output1"}'))
    #configuration of:  actionLink2
    app._private.page_widgets.append(json.loads('{"widgetId":"actionLink2","name":"actionLink","value":{"label":"Click for navigate",
    "textStyle":"ux-body1","textColor":"blue","eventType":"navigation"},"position":"#actionLink2"}'))
    #configuration of:  progressBar1
    app._private.page_widgets.append(json.loads('{"widgetId":"progressBar1","name":"progressBar",
    "value":{"width":150,"showLabel":true,"labelPosition":"right","percentage":42,"progressColor":"orange","labelColor":"grey-p3",
    "centered":false,"size":"small"},"position":"#progressBar1"}'))
 
 
    return {"layout": "pages/layouts/Page/", "pageTitle": "Page", "widgets": app._private.page_widgets}
 
 
def handler():
 
    if app.event.source == "actionLink1": #Refresh Event 1
        app._private.targets = []       # variable defined by the Design Studio Editor to store target widgets
        app._private.targets.append(json.loads('{"widgetId":"output1","name":"output","value":{"label":"Simple text",
        "class":"","textStyle":"ux-body1","textColor":"grey-p3"},"position":"#output1"}'))
 
        #setting bindings
        app._private.targets[0]["value"]["label"] = app.current_session.user
        return {"widgets": app._private.targets}
 
    return {}
 
 
def router():
 
    if app.event.source == "actionLink2": #Navigation Event 1
        app._private.navigation_context = {}    # variable defined by the Design Studio Editor to store Navigation Context
        return {"targetPage": "Page2", "navigationContext": app._private.navigation_context}
 
    return {}
 
 
def update():
    app.page_variables.increase = app.page_variables.increase + 1
 
    app._private.targets = []   # variable defined by the Design Studio Editor to store target widgets
    app._private.targets.append(json.loads('{"widgetId":"progressBar1","name":"progressBar",
    "value":{"width":150,"showLabel":true,"labelPosition":"right","percentage":42,"progressColor":"orange",
    "labelColor":"grey-p3","centered":false,"size":"small"},"position":"#progressBar1"}'))
 
    #setting bindings
    app._private.targets[0]["value"]["percentage"] = app.page_variables.increase
    return {"widgets": app._private.targets}
 
 
def exit():
    print("User "+ app.current_session.user + " exits from the page")