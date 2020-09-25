
election=read.csv("orange_sprite.csv")
election$gender_code=as.character(as.factor(election$gender_code))
election_sub=subset(election,Race=="Asian"|Race=="White"|Race=="Black")

#metrics by race (i.e. vote likelihood)
group_avg_race=election_sub %>% 
  group_by(election_sub$Race) %>%
  summarise_all("mean")

group_avg_race1=as.data.frame(group_avg_race)
colnames(group_avg_race1)[1]="race"

#plot vote likelihood by race 
color="orange"
ggplot(group_avg_race1,aes(x=race,y=vote_lh,fill=color))+geom_bar(stat="identity")+
  xlab("Race")+ylab("Vote Likelihood")
