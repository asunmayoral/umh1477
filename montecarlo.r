# Función para estimar por MC
montecarlo = function(x,est='media',a=-Inf,b=+Inf,dec=4,print=TRUE){
  # x es un vector de simulaciones
  # est={'media'/'prob'} para estimar medias o probabilidades
  # P(a<X<=b) probabilidad a calcular (valores por defecto +-infinito)
  # dec = decimales en la visualización del resultado
  # print=TRUE presenta el resultado, FALSE solo devuelve un dataframe

  # comprobación del tipo de estimación solicitada
  if(est=='prob'){
    z = ((a<x) & (x<=b))*1
  }
  else if(est=='media'){
    z=x
  }
  else{
    cat("Solo podemos estimar media y prob")
    break
  }

# número de datos
n=length(z)
estim = round(mean(z),dec)
error = sd(z)*sqrt(n-1)/n
alpha=0.95 # nivel de confianza
z_alpha= qnorm(1-alpha/2)
ic.low = round(estim - z_alpha*error,3)
ic.up = round(estim + z_alpha*error,3)
if(print==TRUE){
  cat('
 Estimación MC = ',estim,'[',ic.low,',',ic.up,']
')
}
df=c(estim,ic.low,ic.up)
return(df)
}