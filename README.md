# Imperial-Capstone
Linear Programming Code for Imperial Capstone

## Instructions for the Project

With an increase in the elderly population, home medical alert systems are a multi-million pound and dollar business in Europe and the USA. These monitoring systems are often used by people with pre-existing medical conditions or post-operative patients who have returned home but are still required to be monitored in case of infection or other complications. The devices are usually worn by the user and send a signal to a specialized medical alert company that passes on any medical alerts from the devices to doctors and hospitals. 
There are many such devices out there in the marketplace and prices range from hundreds to tens of pounds or dollars depending on the type of monitoring being performed by the device. As such, competition in the market is strong, so companies must be competitive to sell their products. 
You are a research and development manager working in a small electronic company that manufactures a simple medical alert system called HeartSafe, which monitors users with heart conditions at home. If the user starts to display heart problems, such as Arrhythmia or fibrillation, the device signals an alert to the user and sends an emergency alert to a medical alert company. 
While this is a good selling and useful device, you believe that you can make a similar more enhanced device that could serve some members of the public better, while at the same time making more profit for the company. 
With your strong background in R&D and business management, you come up with the idea of a second device, which you call the SmartAlert system. This system also monitors respiration rate, blood oxygen, and blood pressure, so it can detect the early onset of a variety of medical issues including fainting, COPD, hypoxia as well as heart problems. 
After doing a technical and business evaluation of the idea you decide to do a business proposal presentation to the upper management in the company to try to convince them of the merits of producing this new product. In your business proposal, you will have to show if it is more profitable to just replace the present HeartSafe device with the new SmartAlert device or if you should produce some combination of the two different devices. 
For simplicity, we will refer to these device types as 
**Type X for the HeartSafe device and Type Y for the SmartAlert device.** 
With the machinery you have right now, you see that you could make 200 Type X devices per hour, but you could also make 140 Type Y devices per hour on the same machinery. After some calculations, you note that the profit per Type X device is $25 and the profit per Type Y device is $30. You have also discovered that in a week you can create a maximum of 6000 Type X devices and 4000 Type Y devices because of the person-hours and capacity of specialists available to you. 
Note that you have only 40 hours per week to produce them. 

## Method Used

Linear Programming was used to calculate the Optimal Profit based on the constraints given. 
The Above expression can be broken down into the following mathematical expressions

### Expected Profit

25* x + 30 * y = Profit

### Limitations/constraints equations

#### Time per week Constraints

x/200 + y/140 = 40

#### Production Max Cobstarints for both Type x and Type Y

x ≤ 6000

y ≤ 4000
