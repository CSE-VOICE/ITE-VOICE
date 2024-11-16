# VOICE: A Multimodal Routine Recommender for Any Circumstances
> 한양대학교 2024-2 소프트웨어공학/인공지능및응용 프로젝트 (SWE/ITE Project in Hanyang Univ. 2024-2)

## Proposal
We propose a multimodal AI service for recommending suitable smart home routines to users in any circumstances. To do so, we divided our service into two phases.

In the first phase, we plan to make a generative AI model that returns optimal smart home routines for users in any situations using text. In order to train this, we will derive tens of thousands of data with the format (circumstance: routine) using ChatGPT. To guarantee the quality of recommendation in any situations, we will construct dataset with various circumstances, even if it does not make much sense. After gathering dataset, we will train the model using transfer learning with transformer based GPT-2 or Llama(will be decided later). This will be done by using hugging face library.

In the second phase, we will expand our domain from text to speech with SKT NUGU. We plan to implement speech recognition with verbal details such as emotion, tone, and speed of voice using existing libraries. Therefore, we can recommend optimal routine for verbal requests. Furthermore, our service keeps track of users’ conversation so that it can capture context, keywords and extra information like urgency and provide optimal smart home routine.

## Contributors
| Role | Name | Organization | Email |
|------|-------|-------|-------|
| AI Developer | Jaehwi Song | Dept. of Information Systems, College of Engineering, Hanyang University | wotns0319@naver.com |
| Software Developer (Backend) | Hyeonseo Yu | Dept. of Information Systems, College of Engineering, Hanyang University | daina192939@gmail.com |
| Data Engineer | Seungjin Lee | Dept. of Information Systems, College of Engineering, Hanyang University | cookjin@hanyang.ac.kr |
| Software Developer (Frontend) | Hyungrak Choi | Dept. of Information Systems, College of Engineering, Hanyang University | hrgr0711@naver.com |
