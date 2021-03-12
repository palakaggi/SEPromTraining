#plotting to check if at 500, there is a dip CORRESPONDING TO EACH PARAMETRE(to indicate TSS)

lineplt_ex = parameter_map_tss['normalized_params_map'][0]['a']
a = [0 for i in range(976)]
b = [0 for i in range(976)]
c = [0 for i in range(976)]


for i in range(19):
    a=np.add(a,parameter_map_tss['normalized_params_map'][i]['a'])
    b = np.add(b, parameter_map_tss['normalized_params_map'][i]['b'])
    c = np.add(c, parameter_map_tss['normalized_params_map'][i]['g'])

a = [i/19 for i in a]
b = [i/19 for i in b]
c = [i/19 for i in c]
plt.plot(a,label = 'a',color = 'b')
plt.plot(b,label = 'b',color='r')
plt.plot(c,label = 'g',color='g')
# plt.xlim(0,1000)
# plt.ylim(0,1000)

# plt.xticks(ran)
plt.show()