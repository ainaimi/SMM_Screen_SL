packages <- c("dplyr","tidyverse","ggplot2","SuperLearner","VIM","naniar","epiDisplay","tableone","visdat","recipes","resample","caret")
for (package in packages) {
  if (!require(package, character.only=T, quietly=T)) {
    install.packages(package,repos='http://lib.stat.cmu.edu/R/CRAN') 
  }
}

td <- read.csv(file="\\\\136.142.117.70\\Studies$\\Bodnar Abby\\Severe Maternal Morbidity\\Data\\train_2020.05.29.csv", header=TRUE, sep=",")

#Build our preprocessing steps into a blueprint using recipes()

# Removing the variables with the most missing and that we weren't planning on using anyway from the data set
td <- td %>% 
  dplyr::select(-c("n_mom", "gwgkg", "bmiprepreg", "mobrpwt_kg", "X", "deliverytime", "momdcdate", "deliverytime", "ch_new", "lmpdate","startprg","caseno","n_pregs"))

# Also removing duplicate variables -- original versions of variables we already recoded
td <- td %>% 
  dplyr::select(-c("mdelvlac","fdelcsxn", "M_smm12",
                   "fdelforc","menddiab","mendthy","mhemanem","mgurenal","abort","grav","para","mlabindu",
                   "lastpregoutcome","infdcdisp","fdelmal", "apgar1_cat", "apgar5_cat"))

#Visualizing missing with a heat map
sum(is.na(td))

td %>% is.na() %>% 
  reshape2::melt() %>% 
  ggplot(aes(Var2, Var1, fill=value)) +
  geom_raster() +
  coord_flip() +
  scale_y_continuous(NULL, expand=c(0,0)) +
  scale_fill_grey(name="", labels=c("Present","Missing"))+
  theme(axis.text.y = element_text(size=4))



# Now we can start building a recipe to do what we want 

str(td, list.len=ncol(td))

# I think I want to make all 0/1 integers as well as some others factors

# First make sure the variables we want to treat as numeric actually are numeric
td$birthweight <- as.numeric(td$birthweight)
td$delwksgt <- as.numeric(td$delwksgt)
td$momiculos <- as.numeric(td$momiculos)
td$niculos <- as.numeric(td$niculos)
td$los <- as.numeric(td$los)
td$ch_smmtrue <- as.numeric(td$ch_smmtrue)
td$momage <- as.numeric(td$momage)
td$apgar1 <- as.numeric(td$apgar1)
td$apgar5 <- as.numeric(td$apgar5)

# Remaining integers: convert to factors so you can do proper mode imputation
int_names <- td %>% select_if(is.integer) %>% colnames
#These all look good -- I want to convert all of these integers to factors, then back.
td[,int_names] = data.frame(apply(td[int_names],2,as.factor))

#Now look again and make sure every variable is the way you want it
str(td, list.len=ncol(td))

# Now start building the recipe
# Keep near-zero variance predictors in 
# Mode imputation has to come first or else it throws an error because I think some of the SMM variables with one level are kept in
momi_recipe <- recipe(ch_smmtrue ~., data = td)
momi_recipe_steps <- momi_recipe %>% 
  step_modeimpute(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors(), -all_outcomes()) %>%   
  step_meanimpute(all_numeric(), -all_outcomes()) %>% 
  step_YeoJohnson(all_numeric(), -all_outcomes()) %>% 
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(),-all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = FALSE)

prep_momi <- prep(momi_recipe_steps, training = td)
train_momi <- bake(prep_momi, new_data = td)

# Here we can actually look at which zero-variance predictors were kicked out
prep_momi$steps[[2]]

# Save training data
saveRDS(train_momi, file = "\\\\136.142.117.70\\Studies$\\Bodnar Abby\\Severe Maternal Morbidity\\Data\\baked_train_momi_20200615.rds")


# Now, we want to see if we can figure out a way to save an additional "baked" training data set that incorporates splines on continuous variables (for gams)
momi_recipe <- recipe(ch_smmtrue ~., data = td)
momi_recipe_splines <- momi_recipe %>% 
  step_modeimpute(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors(), -all_outcomes()) %>%   
  step_meanimpute(all_numeric(), -all_outcomes()) %>% 
  step_YeoJohnson(all_numeric(), -all_outcomes()) %>% 
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(),-all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = FALSE) %>% 
  step_bs(birthweight, delwksgt, momiculos, momhgt, mobadmwt_g, momage, apgar1, apgar5, niculos)

prep_momi_splines <- prep(momi_recipe_splines, training = td)
train_momi_splines <- bake(prep_momi_splines, new_data = td)
