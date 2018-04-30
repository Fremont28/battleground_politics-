election=read.csv("orange_sprite.csv")
#white:0, back:2,asian:4
election$gender_code=as.character(as.factor(election$gender_code))

election_sub=subset(election,Race=="Asian"|Race=="White"|Race=="Black")
#metrics by race (i.e. vote likelihood)
group_avg_race=election_sub %>% #source: https://stackoverflow.com/questions/40947288/in-r-how-to-calculate-mean-of-all-column-by-group?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
  group_by(election_sub$Race) %>%
  summarise_all("mean")
group_avg_race
group_avg_race1=as.data.frame(group_avg_race)
colnames(group_avg_race1)[1]="race"

color="orange"
ggplot(group_avg_race1,aes(x=race,y=vote_lh,fill=color))+geom_bar(stat="identity")+
  xlab("Race")+ylab("Vote Likelihood")






