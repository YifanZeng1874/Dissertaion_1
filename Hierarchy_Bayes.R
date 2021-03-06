library(rjags)
library(runjags)
library(readxl)
library("xlsx")

my_data <- read_excel("data/enteric_viruses.xlsx", sheet = "04VZ metadata_707_long_edited_1")


#virus variables
deep_sequence <- my_data[, seq(56,87)]
common_virus <- -(deep_sequence[,c(28,23,31,18,21,20)]-2)
uncommon_virus <- -(deep_sequence[,-c(28,23,31,18,21,20,32)]-2)
unclass_virus <- -(deep_sequence[,32]-2)


common_virus <- cbind(common_virus,common_count = colSums(t(common_virus)))
uncommon_virus <- cbind(uncommon_virus,uncommon_count = colSums(t(uncommon_virus)))

#binary variables
single_common <- as.numeric(common_virus$common_count > 0 & uncommon_virus$uncommon_count == 0)
single_uncommon <- as.numeric(common_virus$common_count == 0 & uncommon_virus$uncommon_count > 0)
between_virus <- as.numeric(common_virus$common_count > 0 & uncommon_virus$uncommon_count > 0)


#remove 'OTHER WS'
data <- my_data[,c(3,9,17,seq(31,36),38,39,40,88)]

#impute
data$ContactDiar[data$ContactDiar == 9] <- 2

#coersion
data$is_coinf[data$is_coinf == 'NA'] <- 0
data$is_coinf <- as.numeric(data$is_coinf)

#as.numeric
data$Gender<- data$Gender - 1  #0:male  1:female


data$Tap <- as.numeric(data$Tap) #0:False 1:True
data$Well <- as.numeric(data$Well)
data$Rain <- as.numeric(data$Rain)
data$River <- as.numeric(data$River)
data$Pond <- as.numeric(data$Pond)
data$Bottled <- as.numeric(data$Bottled)

data$ContactDiar <- data$ContactDiar - 1 #0:Yes 1:No
data$KeepAnimal <- data$KeepAnimal - 1  #0:Yes 1:No
data$KillingAnimal <- data$KillingAnimal - 1
data$EatCookRawMeat <- data$EatCookRawMeat - 1

#
#data$ContactDiar <- factor(data$ContactDiar, labels = c('Contact', 'NotContact'))
#data$Gender <- factor(data$Gender, labels = c('male', 'female'))
#data$KeepAnimal <- factor(data$KeepAnimal, labels = c('Keep', 'NotKeep'))
#data$KillingAnimal <- factor(data$KillingAnimal, labels = c('Kill', 'NotKill'))
#data$EatCookRawMeat <- factor(data$EatCookRawMeat, labels = c('Eat', 'NotEat'))


data <- as.data.frame(cbind(single_common,single_uncommon,between_virus,data,site = my_data$SiteRecruitment))

data$site[data$site == 2] <- 1
data$site[data$site == 4] <- 2
data$site[data$site == 5] <- 3
data$site[data$site == 6] <- 4

#Data block
n <- nrow(data) 
nArea<-length(unique(data$site))
enteric.data <- list(n=n, is_coinf = data$is_coinf, 
                     Age = data$Age, Gender = data$Gender, ContactDiar = data$ContactDiar, 
                     Tap = data$Tap, Well = data$Well, Rain = data$Rain, River = data$River, 
                     Pond = data$Pond, Bottled = data$Bottled, KeepAnimal = data$KeepAnimal,
                     KillingAnimal = data$KillingAnimal, EatCookRawMeat = data$EatCookRawMeat,
                     single_common = data$single_common, single_uncommon = data$single_uncommon,
                     site = data$site)

#initial data
num.chains <- 3
enteric.inits <- function(){
  list(beta0 = rnorm(1,0,10), 
       beta.Age = rnorm(1,0,10), 
       beta.Gender = rnorm(1,0,10),
       beta.ContactDiar = rnorm(1,0,10),
       beta.Tap = rnorm(1,0,10),
       beta.Well = rnorm(1,0,10),
       beta.Rain = rnorm(1,0,10),
       beta.River = rnorm(1,0,10),
       beta.Pond = rnorm(1,0,10),
       beta.Bottled = rnorm(1,0,10),
       beta.KeepAnimal = rnorm(1,0,10),
       beta.KillingAnimal = rnorm(1,0,10),
       beta.EatCookRawMeat = rnorm(1,0,10),
       beta.single_common = rnorm(1,0,10),
       beta.single_uncommon = rnorm(1,0,10),
       sigma.Area = runif(1,0,0.1))}

#Model
#is_coinf ~ beta0 + (Age - mean(Age)) + Gender + ContactDiar + Tap +``````

enteric.model <-"model{
#likelihood
for(i in 1:n){
is_coinf[i] ~ dpois(mu[i])

mu[i] = exp(beta0 + beta.Age*(Age[i]-mean(Age[])) + beta.Gender*Gender[i] + beta.ContactDiar*ContactDiar[i] + 
beta.Tap*Tap[i] + beta.Well*Well[i] + beta.Rain*Rain[i] + beta.River*River[i] + beta.Pond*Pond[i] + 
beta.Bottled*Bottled[i] + beta.KeepAnimal*KeepAnimal[i] + 
beta.KillingAnimal*KillingAnimal[i] + beta.EatCookRawMeat*EatCookRawMeat[i]+
beta.single_common*single_common[i] + beta.single_uncommon*single_uncommon[i] + beta.ar[site[i]])}

for(j in 1:4){
beta.ar[j]~ dnorm(0,tau.Area)
}


#prior
beta0 ~ dnorm(0,0.0001)
beta.Age ~ dnorm(0,0.0001) 
beta.Gender ~ dnorm(0,0.0001)
beta.ContactDiar ~ dnorm(0,0.0001)
beta.Tap ~ dnorm(0,0.0001)
beta.Well ~ dnorm(0,0.0001)
beta.Rain ~ dnorm(0,0.0001)
beta.River ~ dnorm(0,0.0001)
beta.Pond ~ dnorm(0,0.0001)
beta.Bottled ~ dnorm(0,0.0001)
beta.KeepAnimal ~ dnorm(0,0.0001)
beta.KillingAnimal ~ dnorm(0,0.0001)
beta.EatCookRawMeat ~ dnorm(0,0.0001)
beta.single_common ~ dnorm(0,0.0001)
beta.single_uncommon ~ dnorm(0,0.0001)

#Hyperparameters
tau.Area<- pow(sigma.Area,-2)
sigma.Area ~ dunif(0,10)

}"


# Running JAGS

enteric.res.A <- jags.model(file = textConnection(enteric.model),
                            data = enteric.data, n.chains=3,
                            inits = enteric.inits, quiet = TRUE)
update(enteric.res.A, n.iter=5000)
enteric.res.B <- coda.samples(enteric.res.A,
                              variable.names=c("beta0","beta.Age",
                                               "beta.Gender","beta.ContactDiar",
                                               "beta.Tap","beta.Well","beta.Rain",
                                               "beta.River","beta.Pond","beta.Bottled",
                                               "beta.KeepAnimal","beta.KillingAnimal",
                                               "beta.EatCookRawMeat","beta.single_common",
                                               "beta.single_uncommon",
                                               "beta.ar","sigma.Area","is_coinf"), n.iter=100000)


# Getting DIC
#enteric.res.DIC <- dic.samples(model=enteric.res.A,
#                                 n.iter = 100000,type = "pD")

# Joinning all the chains in one data.frame
enteric.output <- do.call(rbind.data.frame, enteric.res.B)

# Checking the chains
#mcmcplots::mcmcplot(enteric.res.B, parms
#                    = c("beta0", ))

# Checking the effective sample size
effectiveSize(enteric.res.B) # (all > 24000)


#> effectiveSize(enteric.res.B) # (all > 24000)
#beta.Age         beta.Bottled     beta.ContactDiar  beta.EatCookRawMeat 
#96969.0080            2885.1685            2875.8420           94695.8164 
#beta.Gender      beta.KeepAnimal   beta.KillingAnimal            beta.Pond 
#72911.9780           77717.5693            1501.0995           20922.4067 
#beta.Rain           beta.River             beta.Tap            beta.Well 
#7586.5027            2867.1622            2780.7280            3965.9897 
#beta.single_common beta.single_uncommon                beta0 
#59151.3017           65330.5426             706.2372 
 
# Checking the Gelman-Ruben-Brooks statistic
gelman.diag(enteric.res.B) # (all = 1)


#Potential scale reduction factors:

#  Point est. Upper C.I.
#beta.Age                     1          1
#beta.Bottled                 1          1
#beta.ContactDiar             1          1
#beta.EatCookRawMeat          1          1
#beta.Gender                  1          1
#beta.KeepAnimal              1          1
#beta.KillingAnimal           1          1
#beta.Pond                    1          1
#beta.Rain                    1          1
#beta.River                   1          1
#beta.Tap                     1          1
#beta.Well                    1          1
#beta.single_common            1       1.00
#beta.single_uncommon          1       1.00
#beta0                        1          1

#Multivariate psrf
#1

# Summary
enteric.summ <- summary(enteric.res.B) # all Ts-SE < 5% SD

round(mean(exp(enteric.output$beta0)),3) 

#summary
summary(enteric.res.B)


round(mean(exp(enteric.output$beta0)),3)  #0.806   average0.755
round(mean(exp(enteric.output$beta.Age)),3)  #0.986  age 1 unit  co_in decrease


round(mean(exp(enteric.output$beta.Gender)),3) #0.924  0:male 1:female  
#less for female   butcher for male


#sum(data$Gender == 1) 359 female
#sum(data$Gender == 0) 348 male

#0:Yes 1:No
round(mean(exp(enteric.output$beta.ContactDiar)),3) # 1.162  (disagree unbalance)
#no contact, more co_in
#sum(data$ContactDiar == 9)  14
#sum(data$ContactDiar == 1)  18   yes
#sum(data$ContactDiar == 2)  675  no


round(mean(exp(enteric.output$beta.EatCookRawMeat)),3) # 0.972 (disagree)
#sum(data$EatCookRawMeat == 1)  185  no   no eat, more co_in
#sum(data$EatCookRawMeat == 0)  522  yes

round(mean(exp(enteric.output$beta.KeepAnimal)),3) #0.999
#summary(data$KeepAnimal)    no keep, more co_in  (disagree  coincident)
#Keep NotKeep 
#404     303 


round(mean(exp(enteric.output$beta.KillingAnimal)),3) #1.243

#summary(data$KillingAnimal)  no kill, more co_in  (disagree  unbalance)
#Kill NotKill 
#14     693 


#0:False 1:True
round(mean(exp(enteric.output$beta.Bottled)),3) #1.072  more coin (both bottle and other?)
#sum(data$Bottled == 0)  527
#sum(data$Bottled == 1)  180

round(mean(exp(enteric.output$beta.Pond)),3) #0.857   less coin (disc unbalance?)
#sum(data$Pond == 0)  704
#sum(data$Pond == 1)  3


round(mean(exp(enteric.output$beta.Rain)),3)  #0.91  less coin (disc unbalance)
#sum(data$Rain == 0)  694
#sum(data$Pond == 1)  13

round(mean(exp(enteric.output$beta.River)),3)  #1.096 more coin  (agree)
#sum(data$River == 1)  172
#sum(data$River == 0)  535

round(mean(exp(enteric.output$beta.Tap)),3)  #1.197  more coin  (agree)
#sum(data$Tap == 1)  330
#sum(data$Tap == 0)  377

round(mean(exp(enteric.output$beta.Well)),3)  # 1.323  more coin  (not sure)
#sum(data$Well == 0)   683
#sum(data$Well == 1)   24

round(mean(exp(enteric.output$beta.single_common)),3)  #1.303    more coin
round(mean(exp(enteric.output$beta.single_uncommon)),3)  #2.052 much more coin

