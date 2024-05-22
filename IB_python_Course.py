import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.optimize import minimize

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Imperial Business Analysis - Capstone Project")
st.divider()

feasibilitycolor = st.sidebar.color_picker("Choose a color for your feasibility region", '#282828')
optimalmarkercolor = st.sidebar.color_picker("Choose a color for marking Optimal Point", '#229954')
st.sidebar.markdown("\n **Choose Scenarios Simulation**")
scen1 = st.sidebar.checkbox(" Focus Production more on a Particular Device")
if scen1:
	choicescen1 = st.sidebar.radio("Choose Superior Device",["Type X", "Type Y"], horizontal= True)
	diffechoice = st.sidebar.radio("Choose between Minimum or Maximum Difference",["Minimum difference", "Maximum difference"], horizontal= True)
	proddiff = st.sidebar.number_input("What should be the {0} between Products".format(diffechoice), value=None, placeholder="Type a number...")
	st.sidebar.divider()

scen2 = st.sidebar.checkbox(" Produce Just one type of Device(Either X or Y)")
if scen2:
	choicescen2 = st.sidebar.radio("Choose Only Device to Produce",["Type x", "Type y"], horizontal= True)


profit_x = 0
profit_y = 0
prodrate_x = 0
prodrate_y = 0
Amount_x = 0
Amount_y = 0
time_lim = 0

choice = st.radio("Choose either running the capstone demo values or enter personal values", ('Demo: Use Capstone Values', "Enter Personal Values")) 

if choice == "Demo: Use Capstone Values":
	profit_x = 25
	profit_y = 30
	prodrate_x = 200
	prodrate_y = 140
	Amount_x = 6000
	Amount_y = 4000
	time_lim = 40
	st.divider()
elif choice =="Enter Personal Values":
	st.subheader("Profit per Device")
	st.markdown("**Enter the epected or estimated Profit per Device**")
	st.write("Type X - Profit per Device")
	profit_x = st.number_input('Enter the Profit per device for Device Type x')
	st.write("Type Y - Profit per Device")
	profit_y = st.number_input('Enter the Profit per device for Device Type y')
	st.divider()

	st.subheader("Production Limit and Constraints")
	st.markdown("**Production Rate**")
	st.write("Production rate per hour for Device Type x")
	prodrate_x = int(st.number_input('How many goods of type x can be produced in an hour'))
	st.write("Production rate per hour for Device Type y")
	prodrate_y = int(st.number_input('How many goods of type y can be produced in an hour'))
	st.markdown("**Time per week Limit**")
	st.write("Total Allocated time per week")
	time_lim = int(st.number_input('How many hours do you have in a week?'))

	st.markdown("**Maximum Production Capacity**")
	st.write("Production Capacity for Device- Type x")
	Amount_x = int(st.number_input('What is the Maximum Production for Type X'))
	st.write("Production Capacity for Device- Type y")
	Amount_y = int(st.number_input('What is the Maximum Production for Type Y'))


st.markdown("**Production Profit Equations**")
st.write(" Profit Equation:  {0} * x + {1} * y".format(profit_x, profit_y))
st.markdown("**Constraints and Limit Equations**")
st.write(" Time Limit Constraints: x/{0} + y/{1} = {2}".format(prodrate_x, prodrate_y, time_lim))
st.write(" Max Production Limit: x <= {0}".format(Amount_x))
st.write(" Max Production Limit: y <= {0}".format(Amount_y))

st.divider()

type_x = np.linspace(0, Amount_x, 1000) # generate 1000 equaly spaced values between 0 and 6000 
type_y = np.linspace(0, Amount_y, 1000 ) # generate 1000 equaly spaced values between 0 and 4000



fig2 = plt.figure()
plt.figure(figsize= (15, 10))
# Plot the constraint equation
plt.plot(type_x, prodrate_y*(time_lim - type_x/prodrate_x), label="x/{0} + y/{1} = {2} ---> C1".format(prodrate_x, prodrate_y, time_lim))
# Plot the constraint lines
plt.axvline(x=Amount_x, color='k', linestyle='--', label="x = {0}".format(Amount_x))
plt.axhline(y=Amount_y, color='brown', linestyle='--', label="y = {0}".format(Amount_y))
# # limits
plt.ylim(0)
plt.xlim(0)
plt.title('Graph of Constraints Showing Feasible Area')
plt.xlabel('Type X', fontsize = 20)
plt.ylabel('Type Y', fontsize = 20)

# fill in the feasible region
plt.fill_between(type_x, # regular item 1 
                  np.minimum(prodrate_y*(time_lim - type_x/prodrate_x), Amount_y), # choose the minimum between x2 from two eqns
                 where = type_x >= 0, # defines any condition that might need to considered
                 color =feasibilitycolor, # shaded region
                 alpha = 0.50) # transparency level
# # legend
plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
st.subheader("Marking out the Feasible Region on Production Graph")
st.pyplot()
st.divider()

# Construct parameters
profit = np.array([profit_x, profit_y]) #comes from the coefficients of the profit equation
if prodrate_x != 0 or prodrate_y != 0:
	# Inequality constraints
	limitations = np.array([[1/prodrate_x, 1/prodrate_y], # first row comes from constriants 1
	                     [1, 0], [0, 1]])  # second row comes from constraints 2

	constraints = np.array([time_lim, Amount_x, Amount_y]) # max amount of constraint


	# Solve the problem
	# we put a negative sign on the objective as linprog does minimization by default
	production = linprog(-profit, A_ub = limitations, b_ub = constraints, method='revised simplex')

	st.subheader("Optimal Solution")
	st.markdown("**The Optimal Production Value based on Constraints given earlier is;**")
	st.write('Profit:', abs(round(production.fun, ndigits=2)))
	st.write('Amount of Type X:', round(production.x[0]))
	st.write('Amount of Type Y:', round(production.x[1]))
	st.write('Status:', production.message)

	fig3 = plt.figure()
	plt.figure(figsize= (15, 10))
	# Plot the constraint equation
	plt.plot(type_x, prodrate_y*(time_lim - type_x/prodrate_x), label="x/{0} + y/{1} = {2} ---> C1".format(prodrate_x, prodrate_y, time_lim))
	plt.plot(production.x[0], production.x[1], marker="o", markersize=20, markeredgecolor=optimalmarkercolor, markerfacecolor=optimalmarkercolor)
	plt.text(production.x[0], production.x[1], f'Optimal x={round(production.x[0])}\nOptimal y={round(production.x[1])}\nProfit={abs(round(production.fun, ndigits=2))}', fontsize=12, ha='left')
	# Plot the constraint lines
	plt.axvline(x=Amount_x, color='k', linestyle='--', label="x = {0}".format(Amount_x))
	plt.axhline(y=Amount_y, color='brown', linestyle='--', label="y = {0}".format(Amount_y))
	# # limits
	plt.ylim(0)
	plt.xlim(0)
	plt.title("Production Graph Showing Optimal Point marked in red")
	plt.xlabel('Type X', fontsize = 20)
	plt.ylabel('Type Y', fontsize = 20)

	# fill in the feasible region
	plt.fill_between(type_x, # regular item 1 
	                  np.minimum(prodrate_y*(time_lim - type_x/prodrate_x), Amount_y), # choose the minimum between x2 from two eqns
	                 where = type_x >= 0, # defines any condition that might need to considered
	                 color =feasibilitycolor, # shaded region
	                 alpha = 0.50) # transparency level
	# # legend
	plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)

	st.subheader("Marking out the Feasible Region on Production Graph")
	st.pyplot()
else:
	st.warning("Enter the Profit and Constraints Values above or Select the Demo button to run pre-fixed numbers")
st.divider()
#creating a Profit Prognosis Chart and then plot
st.subheader("Profit Prognosis Plot")
st.markdown("**Profit Generation according to variation in production at different amount of devices**")
chart_y = list(np.linspace(0,Amount_y,5, dtype=int)) #taking equally spaced values from 0 up to the maximum value

#using function to prevent an over spill of values past the limit
def get_x(y):
    if prodrate_x *(time_lim - y//prodrate_y) > Amount_x:
        return Amount_x
    else:
        return prodrate_x *(time_lim - y//prodrate_y)
    
chart_x = [get_x(j) for j in chart_y]
chart_profit = [round(profit_x*chart_x[l] + profit_y*chart_y[l], 2) for l in range(len(chart_x))]
#adding optimised points in the graph
chart_x.append(round(production.x[0]))
chart_y.append(round(production.x[1]))
chart_profit.append(abs(round(production.fun,2)))
chart_dict = {'Type X': chart_x, 'Type Y': chart_y, 'Profit': chart_profit}
df = pd.DataFrame(chart_dict)
df2 = df.sort_values(by=['Type Y','Type X'])
st.dataframe(df2.style.highlight_max('Profit',color = 'lightgreen', axis = 0))

st.divider()

if st.button("Run Scenario Simulations"):
	if scen1:
		if proddiff is not None:
			if choicescen1 == "Type X":
				variablechanger = [-1, 1]
			elif choicescen1 == "Type Y":
				variablechanger = [1,-1]
			if diffechoice == "Maximum difference":
				variablechanger = [i * -1 for i in variablechanger]
				proddiff = -1* proddiff

			limitationsscen1 = np.array([[1/prodrate_x, 1/prodrate_y], # first row comes from constriants 1
	                     [1, 0], [0, 1], variablechanger])  # second row comes from constraints 2
			constraintsscen1 = np.array([time_lim, Amount_x, Amount_y, proddiff] )
			productionscen1 = linprog(-profit, A_ub = limitationsscen1, b_ub = constraintsscen1, method='revised simplex')

			st.subheader("Scenario 1")
			st.markdown("**Based on scenario selected and a {0} in product of {1};**".format(diffechoice, abs(proddiff)))
			st.write('Profit:', abs(round(productionscen1.fun, ndigits=2)))
			if abs(round(productionscen1.fun, ndigits=2)) < abs(round(production.fun, ndigits=2)):
				st.write("This simulation gives a profit les than the optimal profit")
			st.write('Amount of Type X:', round(productionscen1.x[0]))
			st.write('Amount of Type Y:', round(productionscen1.x[1]))
			st.write('Status:', productionscen1.message)
		else:
			st.error("Enter a value for the product difference in the first Scenario by the sidebar")
	st.divider()
	if scen2:
		if choicescen2 == "Type x":
			constraintsscen2 = np.array([time_lim, Amount_x, 0] )
		elif choicescen2 == "Type y":
			constraintsscen2 = np.array([time_lim, 0, Amount_y] )

		limitationsscen2 = np.array([[1/prodrate_x, 1/prodrate_y], # first row comes from constriants 1
	                     [1, 0], [0, 1]])  # second row comes from constraints 2
		productionscen2 = linprog(-profit, A_ub = limitationsscen2, b_ub = constraintsscen2, method='revised simplex')

		st.subheader("Scenario 2")
		st.markdown("**If we focus on producing just Device {0};**".format(choicescen2))
		st.write('Profit:', abs(round(productionscen2.fun, ndigits=2)))
		if abs(round(productionscen2.fun, ndigits=2)) < abs(round(production.fun, ndigits=2)):
			st.write("This simulation gives a profit les than the optimal profit")
		st.write('Amount of Type X:', round(productionscen2.x[0]))
		st.write('Amount of Type Y:', round(productionscen2.x[1]))
		st.write('Status:', productionscen2.message)
