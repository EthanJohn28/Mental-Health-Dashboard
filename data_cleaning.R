library(dplyr)
library(readr)
library(ggplot2)
library(scales)
df <- read.csv("/Users/ethanjohn/Desktop/Data Science/Projects/MentalHealthDashboard/Students Social Media Addiction.csv")

glimpse(df)

#Check missing values
missing_vals = sum(is.na(df))
print(missing_vals)

#Create new derived features (feature engineering)


df_2 <- df %>%
  mutate(

        # Normalized base components
    Usage_norm = rescale(Avg_Daily_Usage_Hours),
    Sleep_norm = rescale(Sleep_Hours_Per_Night),
    Conflicts_norm = rescale(Conflicts_Over_Social_Media),
    Sleep_Deficit_norm = rescale(pmax(0, 8 - Sleep_Hours_Per_Night)),
    

    #Balance_Score = Sleep_Hours_Per_Night - Avg_Daily_Usage_Hours,

    #Usage_Sleep_Ratio = Avg_Daily_Usage_Hours / Sleep_Hours_Per_Night,

    #Usage_Efficiency_Level = case_when(
        #Usage_Sleep_Ratio < 0.3 ~ "Healthy",
        #Usage_Sleep_Ratio < 0.6 ~ "Moderate",
        #TRUE ~ "Unhealthy"
    #),
    
    #Platform_Usage = paste(Most_Used_Platform, "-", Usage_Level),

    #Lifestyle_Balance_Score = (Sleep_Hours_Per_Night - Avg_Daily_Usage_Hours) * Mental_Health_Score,

    # Note, maybe use ML to predict factors that influence affects academic performance
    Usage_Hrs_Level = case_when(
        Avg_Daily_Usage_Hours < 2 ~ "Low",
        Avg_Daily_Usage_Hours < 5 ~ "Medium",
        TRUE ~ "High"
    ),

    Sleep_Deficit = 8 - Sleep_Hours_Per_Night,

    Platform_Categories = case_when(
      Most_Used_Platform %in% c("YouTube", "TikTok") ~ "Video-Based",
      Most_Used_Platform %in% c("LinkedIn") ~ "Professional Networking",
      Most_Used_Platform %in% c("Instagram", "Facebook", "VKontakte") ~ "Content-Sharing",
      Most_Used_Platform %in% c("Twitter") ~ "Microblogging",
      Most_Used_Platform %in% c("Snapchat", "WhatsApp", "WeChat", "LINE", "KakaoTalk") ~ "Messaging",
      TRUE ~ "Other"
    ),

    Addicted_Score_Levels = case_when(
      Addicted_Score < 3 ~ "Low",
      Addicted_Score < 6 ~ "Moderate",
      TRUE ~ "High"
    ),

    Wellbeing_Score = Mental_Health_Score + Sleep_Hours_Per_Night - Addicted_Score,

    Relationship_Status_Encoded = case_when(
      Relationship_Status == "Single" ~ 1,
      Relationship_Status == "In Relationship" ~ 2,
      Relationship_Status == "Complicated" ~ 3

    ),

    Affects_Academic_Performance_Encoded = case_when(
      Affects_Academic_Performance == "Yes" ~ 1,
      Affects_Academic_Performance == "No" ~ 0

    ),

    Academic_Level_Encoded = case_when(
      Academic_Level == "High School" ~ 1,
      Academic_Level == "Undergraduate" ~ 2,
      Academic_Level == "Graduate" ~ 3

    ),

    Gender_Encoded = case_when(
      Gender == "Male" ~ 1,
      Gender == "Female" ~ 2
    ),


    Academic_Risk_Index = rowMeans(cbind(
      rescale(Addicted_Score),
      rescale(Avg_Daily_Usage_Hours),
      1- rescale(Sleep_Hours_Per_Night),
      1- rescale(Mental_Health_Score)
    )),

    Addicted_Score_Pred =
  0.70 * Avg_Daily_Usage_Hours +
  0.15 * Conflicts_Over_Social_Media +
  0.15 * pmax(0, 8 - Sleep_Hours_Per_Night)
  ,

    MH_Score_Pred =
      0.5 * Sleep_Hours_Per_Night +
      0.3 * (Avg_Daily_Usage_Hours) +
      0.2 * (Conflicts_Over_Social_Media)
  )

head(df_2)
write.csv(df_2, "/Users/ethanjohn/Desktop/Data Science/Projects/MentalHealthDashboard/Students_Social_Media_Addiction_FE.csv", row.names = FALSE)
