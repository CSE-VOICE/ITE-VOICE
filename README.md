# VOICE: A Multimodal Routine Recommender for Any Circumstances
> 한양대학교 2024-2 소프트웨어공학/인공지능및응용 프로젝트 (SWE/ITE Project in Hanyang Univ. 2024-2)

## Members
| Role | Name | Organization | Email |
|------|-------|-------|-------|
| AI Developer | Jaehwi Song | Dept. of Information Systems, College of Engineering, Hanyang University | wotns0319@naver.com |
| Software Developer (Backend) | Hyeonseo Yu | Dept. of Information Systems, College of Engineering, Hanyang University | daina192939@gmail.com |
| Data Engineer | Seungjin Lee | Dept. of Information Systems, College of Engineering, Hanyang University | cookjin@hanyang.ac.kr |
| Software Developer (Frontend) | Hyungrak Choi | Dept. of Information Systems, College of Engineering, Hanyang University | hrgr0711@naver.com |


## Introduction
We propose a multimodal AI service that recommends optimal smart home routines tailored to diverse user circumstances. Our approach integrates generative AI and multimodal interaction capabilities to deliver context-aware automation. To do so, we divided our service into two phases.

In the initial phase, we developed a generative AI model fine-tuned to deliver optimal smart home routines based on textual descriptions of user circumstances. To train the model, we constructed a robust dataset containing approximately 10,000 examples in the format (situation: routine). The dataset was generated using state-of-the-art generative AI models like ChatGPT, MS Copilot, Google Bard, and Claude. To ensure the recommendations are robust and contextually relevant, we included diverse and even unconventional circumstances in the dataset, expanding its versatility across realistic and imaginative scenarios. Once the dataset was prepared, we fine-tuned a transformer-based model, specifically the paust-t5-chat-large, optimized for Korean conversational contexts.

In the second phase, our service extends from text-based interaction to voice-based commands using SKT NUGU. By integrating speech recognition, the system will process verbal requests while capturing extra verbal details such as emotional tone, urgency, and the speed of speech using IBM Watson API. These voice-derived textual and nuanced details will then be combined to enhance the model’s ability to generate more contextually optimized smart home routine recommendations, further tailoring responses to the user’s specific circumstances. Furthermore, our service keeps track of users’ conversation so that it can capture context, keywords and extra information like urgency and provide optimal smart home routine.

## Datasets

Our dataset comprises approximately 10,000 unique smart home routine recommendations, generated using state-of-the-art generative AI models, including ChatGPT, Microsoft Copilot, Google Bard, and Claude. The aim was to create a comprehensive and diverse dataset of routines that cater to a wide range of situations, from common daily scenarios to highly creative and unexpected contexts.

### Dataset Generation Process

**1. Prompt Design:**

We crafted a detailed prompt designed to elicit smart home routines that directly control appliances, avoiding actions requiring human intervention. This ensured that the dataset aligns with practical use cases for smart home automation. The prompt included:

- A predefined list of smart home devices and their actionable settings.
- Explicit guidelines for valid and invalid routines.
- Examples of diverse scenarios to encourage variety and creativity.

**2. Generative AI Utilization:**

State-of-the-art generative AI models were used to produce routines in a structured JSON format containing:

- **Situation**: A natural language description of the context, including the time and specific details.
- **Routine**: A concise description of device actions adhering to the predefined rules.

**3. Deduplication:**

To ensure the uniqueness of the dataset, we implemented a deduplication process to remove identical situations and their corresponding routines. This step helped maintain variety while preventing redundancy.

**4. Quality Assurance:**

The generated routines were filtered and reviewed to ensure compliance with the following criteria:

- Practical and actionable by smart home devices.
- Free from human-performed actions (e.g., “hang laundry”).
- Well-aligned with the context described in the situation.

**5. Data Export to CSV:**

Once the routines satisfied all the criteria outlined in the previous steps, they were added to a structured CSV file for easier handling and further processing. This step involved:

- Transforming the JSON-formatted routines into a tabular CSV structure with columns for “Situation” and “Routine.”
- Ensuring proper formatting and compatibility for downstream applications, such as training machine learning models or integrating with smart home systems.
- Storing the resulting dataset as `dataset.csv`, which serves as the foundation for subsequent analysis and model training.

### Dataset Highlights

- **Diversity**: The dataset covers a wide range of situations, including both mundane and imaginative contexts. Examples range from “빨래가 너무 많이 쌓였어, 세탁기 돌려야 돼.” to “집안에 벌레가 들어왔어, 조명을 이용해서 유인해보자.”
- **Structure**: Each entry consists of a situation and a corresponding routine in CSV format, enabling seamless integration into machine learning pipelines.
- **Scalability**: The methodology allows for further expansion by generating additional routines using updated prompts or new AI models.

### Sample Entry

```
  "situation": "집에 손님이 오기 전에 분위기를 밝게 만들고 싶어.",
  "routine": "에어컨을 23도로 설정하고 조명을 밝게 조정하며 공기청정기를 켤게요."
```

### Generated Dataset Statistics

- **Total routines requested**: 15,000
- **Unique routines after deduplication**: 10,000
- **Eliminated duplicates & quality checking**: Approximately 50% of initial attempts

This dataset represents a cutting-edge resource for training and fine-tuning AI systems designed to enhance smart home automation, providing realistic and user-centered automation routines tailored to a wide array of situations.

## Methodology

### 1. Data Preprocessing

To ensure consistent and high-quality input for training the model, we conducted thorough preprocessing of the dataset. The primary goal was to standardize the input situation text by embedding it into a predefined template. This helped the model better understand the context and generate consistent and accurate outputs.

- **Input Template Definition**:  
  We defined an input template that outlined specific rules for appliance actions and included examples of both valid and invalid routines. The template ensured that:  
  - Only device-controllable actions were included.  
  - Human-performed actions were excluded.  
  - Device settings were specific and actionable.

- **Implementation**:  
  Using a Python script, each situation in the dataset was transformed using the template, and the preprocessed data was saved in a new CSV file (`preprocessed_dataset.csv`).


### 2. Fine-Tuning the Model

We fine-tuned a pre-trained transformer model to align it with the unique requirements of our smart home automation service.

- **Model Selection**:  
  We used the **paust/pko-chat-t5-large model**, a transformer-based variant of Google Flan-T5 fine-tuned with Korean chat data, as the base model. This choice was made due to its conversational capabilities and suitability for Korean-language tasks.

```
  # Load Tokenizer & Model
  tokenizer = T5TokenizerFast.from_pretrained('paust/pko-chat-t5-large')
  model = T5ForConditionalGeneration.from_pretrained('paust/pko-chat-t5-large')
  model.to(device)
```

- **Challenges and Solutions**:  
  - **Large Model Size**: The model, with over 800M parameters, posed computational challenges for our A10 24GB GPU. To address this, we used the PEFT-based LoRA (Low-Rank Adaptation) approach to reduce the number of trainable parameters while maintaining model performance.  
  - **Memory Optimization**: The model was optimized using bfloat16 precision, reducing memory usage without significant loss in performance.

- **LoRA Configuration**:  
  We targeted specific transformer layers (query, value, key, output) for fine-tuning, significantly reducing computational overhead. The LoRA configuration was as follows:

```
  lora_config = LoraConfig(
      task_type=TaskType.SEQ_2_SEQ_LM,
      r=8,
      lora_alpha=16,
      lora_dropout=0.1,
      target_modules=["q", "v", "k", "o"]
  )
  model = get_peft_model(model, lora_config)
```

### 3. Training Procedure

We trained the model using the preprocessed dataset in a seq2seq learning framework:

- **Data Splitting**:  
  The dataset was split into training (95%) and validation (5%) sets to evaluate model performance during training.

- **Hyperparameters**:  
  - Optimizer: AdamW  
  - Learning Rate: 1e-5  
  - Batch Size: 4  
  - Epochs: 10  

- **Checkpoints**:  
  - Checkpoints were saved every 0.5 epoch to ensure robustness against interruptions and to allow iterative evaluation.  
  - The final model was selected based on validation performance from the saved checkpoints.

```
  # save checkpoint for every 0.5 epoch
  if total_steps % int(len(train_dataset) / (batch_size * 2)) == 0:
      checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{total_steps}")
      model.save_pretrained(checkpoint_path)
```

- **Training Process**:  
  Each batch was fed into the model, and loss values were calculated. Gradients were backpropagated to update trainable parameters, while validation loss was computed at the end of each epoch.

```
  for epoch in range(num_epochs):
      model.train()
      for batch in train_loader:
          outputs = model(input_ids=batch['input_ids'].to(device), labels=batch['labels'].to(device))
          loss = outputs.loss
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
```

### 4. Performance Monitoring

Throughout the training process:

- **Training Loss**: Monitored at regular intervals (every 100 steps (log_interval)) to ensure convergence.
- **Validation Loss**: Evaluated after each epoch to track generalization and avoid overfitting.

```
# Training Loss part
if total_steps % log_interval == 0:
    print(f"Epoch {epoch + 1}, Step {total_steps}, Training Loss: {loss.item()}")
```
```
# Validation part
model.eval()
val_loss = 0
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device) 
        outputs = model(input_ids=input_ids, labels=labels)
        val_loss += outputs.loss.item() * batch['input_ids'].size(0)

    val_loss /= len(val_dataset)
    print(f"Epoch{epoch + 1} : Validation Loss : {val_loss}")
```

### 5. Final Model Selection

After training, the final model was selected based on its validation performance from the last checkpoint. The model was saved for deployment, capable of generating optimized routines for smart home devices based on the input situations.

This methodology ensured that the resulting model was not only computationally efficient but also highly tailored to the specific requirements of the service, offering accurate and actionable routines for various scenarios.

## Evaluation & Analysis
### 1. Training and Validation Loss Trends

The model’s training process was closely monitored through regular measurement of both training and validation losses. Training loss was logged every 100 steps (log interval), while validation loss was recorded at the end of each epoch. The observed trends are summarized in the tables below rounded up to two decimal points.

**Loss Trends Summary**
| Step/Epoch          | Training Loss | Validation Loss |
|----------------------|---------------|-----------------|
| 100                 | 21.03         | -               |
| 200                 | 18.56         | -               |
| 500                 | 13.47         | -               |
| 1000                | 10.25         | -               |
| 2000                | 8.93          | -               |
| 2375 (Epoch 1 End)  | 8.00          | 9.45           |
| 4750 (Epoch 2 End)  | 5.42          | 7.28           |
| 7125 (Epoch 3 End)  | 3.28          | 4.96           |
| 9500 (Epoch 4 End)  | 1.92          | 3.10           |
| 11875 (Epoch 5 End) | 0.87          | 1.32           |
| 14250 (Epoch 6 End) | 0.62          | 0.51           |
| 16625 (Epoch 7 End) | 0.42          | 0.38           |
| 19000 (Epoch 8 End) | 0.10          | 0.09           |
| 21375 (Epoch 9 End) | 0.12          | 0.12           |
| 23740 (Epoch 10 End)| 0.08          | 0.09           |

**Key Observations:**
+ Training loss steadily decreased, reaching 8.00 by the end of epoch 1 and below 1.0 by epoch 5. By the final epoch, the training loss was as low as 0.08, indicating strong convergence.
+ Validation loss followed a similar trend, stabilizing at 0.12 from epoch 8 onwards, demonstrating effective generalization.

### 2. Model Checkpoint Evaluation

Checkpoints were saved every 0.5 epoch, corresponding to 2,375 steps per epoch. Outputs from all 20 checkpoints were evaluated using diverse input situations to determine the best-performing model.

#### Evaluation Process
1. **Sample Input Testing:**
   - The same input was tested across all 20 checkpoints.
   - Outputs were analyzed qualitatively based on relevance, contextual understanding, and practicality for smart home routines.
2. **Selection Criteria:**
   - Contextual Relevance: Alignment with the input situation.
   - Practicality: Feasibility of the routine in a smart home environment.
   - Consistency: Reduced variability in unrelated or illogical outputs.
  
#### Checkpoint Performance Example

**Situation:** *"자다가 깼는데 다시 자고 싶은데 잠이 안 와."*
| **Checkpoint**   | **Step** | **Output**                                                                                 | **Evaluation**                                                                                          |
|------------------|---------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Checkpoint_1187  | 1187    | 멍해져라                                                                                    | Irrelevant and vague.                                                                                  |
| Checkpoint_2374  | 2374    | 로또통                                                                                     | Completely unrelated to the input.                                                                     |
| Checkpoint_3561  | 3561    | 변기에서 온수를 받아서 라면을 끓이는데, 빨래를 넣었어요.                                        | Absurd and illogical.                                                                                  |
| Checkpoint_4748  | 4748    | 변기에서 온수를 받아서 라면을 끓이자.                                                         | Illogical and impractical.                                                                             |
| Checkpoint_5935  | 5935    | 걸레를 닦아내서 화장실을 깨끗하게 만들고, 수건을 정리해 수건 정리를 할게요.                               | Irrelevant and overly specific to cleaning.                                                            |
| Checkpoint_7122  | 7122    | 공기청정기를 켜 공기 순환을 돕고 공기청정기를 작동해 쾌적한 공기를 제공할게요.                            | Reasonable, though redundant.                                                                          |
| Checkpoint_8309  | 8309    | 로봇청소기로 청소를 시작하고, 세탁기로 빨래를 정리할게요.                                         | Acceptable, but lacks contextual relevance to the input.                                                |
| Checkpoint_9496  | 9496    | 창문, 식기 정리 등 청소도구를 사용해 화장실 정리를 할게요.                                         | Unrelated to the input situation. Not Practical.                                                                   |
| Checkpoint_10683 | 10683   | 세탁기를 돌려서 세탁물을 세탁하고, 세탁기 바닥을 청소할게요.                                        | Relevance improving, but still overly focused on cleaning.                                              |
| Checkpoint_11870 | 11870   | 냉장고를 청소하고, 세탁기를 돌려 빨래를 시작할게요.                                              | Lacks direct relevance to the situation.                                                               |
| Checkpoint_13057 | 13057   | 정수기에서 깨끗한 물을 준비해 드릴게요.                                                     | Practical, but lacks contextual alignment with the input.                                               |
| Checkpoint_14244 | 14244   | 공기청정기를 켜고, 조명을 밝게 설정해 마음을 편안하게 유지할게요.                                    | Highly relevant, providing a practical and context-sensitive routine.                                   |
| Checkpoint_15431 | 15431   | 공기청정기로 공기 정화 후 로봇청소기로 바닥 청소할게요.                                            | Strong contextual relevance and practicality.                                                          |
| Checkpoint_16618 | 16618   | 로봇청소기로 바닥을 청소하고, 공기청정기로 공기를 정화할게요.                                        | Balanced response with good contextual understanding.                                                   |
| Checkpoint_17805 | 17805   | 식기세척기를 작동해 식기를 세척하고, 공기청정기를 가동해 신선한 공기를 유지할게요.                          | Practical and contextually aligned, though slightly extraneous.                                         |
| Checkpoint_18992 | 18992   | 조명을 밝게 하고, 로봇청소기로 바닥을 청소해 먼지를 제거할게요.                                      | Practical but slightly disconnected from the input’s context.                                           |
| Checkpoint_20179 | 20179   | 공기청정기를 작동해 실내 공기를 정화하고, 조명을 밝게 조절해 편안한 분위기를 조성할게요.                   | Relevant, with an emphasis on user comfort.                                                            |
| Checkpoint_21366 | 21366   | 스타일러로 옷 정리 후, 공기청정기로 공기 정화할게요.                                              | Practical but unrelated to the input situation.                                                        |
| Checkpoint_22553 | 22553   | 로봇청소기로 바닥을 청소하고, 공기청정기로 실내공기를 개선할게요.                                       | Acceptable but lacked emotional alignment with the input.                                               |
| Checkpoint_23740 | 23740   | TV에서 음악을 들어 긴장을 풀도록 하고, 공기청정기를 가동해 상쾌한 공기를 유지할게요.                           | **Best output**, combining practicality, contextual relevance, and user comfort effectively.           |

#### Final Selection
- Checkpoint_23740 (corresponding to step 23740 and epoch 10) was chosen for deployment.
- This checkpoint consistently produced outputs that were contextually relevant, practical, and tailored to the emotional tone of the input.
  
## Related Work
To develop our multimodal routine recommender system, we leveraged a combination of state-of-the-art tools, libraries, and existing research to ensure efficient implementation and high-quality outcomes.

### 1. Tools and Libraries

- **Hugging Face Transformers**  
  We utilized the `paust/pko-chat-t5-large` model from Hugging Face as the foundation for fine-tuning. Its optimization for Korean conversational contexts made it an ideal choice for generating natural language routines for smart home automation. Hugging Face’s tools for tokenization, training, and model deployment streamlined the process, ensuring compatibility with our dataset.

- **LoRA (Low-Rank Adaptation) via PEFT**  
  To efficiently fine-tune the large transformer model, we implemented LoRA, a Parameter-Efficient Fine-Tuning (PEFT) method. This technique focused on specific layers (`q`, `v`, `k`, `o`) of the transformer, reducing the number of trainable parameters without compromising performance.

- **Speech and Emotion Analysis**  
  - **IBM Watson API**: Used for analyzing nuanced details from voice inputs, such as emotional tone, urgency, and speech speed. These features enhanced the model’s ability to generate context-aware and emotionally aligned smart home routines.  
  - **SKT NUGU**: Integrated for voice command recognition, enabling seamless interaction between users and the smart home system through spoken language.

- **PyTorch for Model Training**  
  PyTorch was the primary framework for implementing and training the model, with utilities like `DataLoader` and `Dataset` simplifying data management and batch processing. The `AdamW` optimizer was employed for training, offering robust performance for large-scale datasets by effectively decoupling weight decay from the optimization step.

- **Data Management with pandas**  
  `pandas` was utilized for preprocessing and managing the dataset, ensuring clean and consistent data inputs for model training.


### 2. Existing Studies and Research

- **Generative AI and Prompt Engineering**  
  Inspired by advancements in generative AI, including ChatGPT, and Claude, our system was designed to focus specifically on actionable, context-aware outputs tailored for smart home environments. Prompt engineering techniques were applied to elicit high-quality outputs from generative AI systems during dataset creation, ensuring practical and device-actionable routines.

- **Smart Home Automation Research**  
  Commercial platforms like Google Home and Amazon Alexa informed our understanding of smart home automation’s current capabilities and limitations. Unlike these systems, our service emphasizes multimodal interaction and emotional nuance to create a more personalized user experience.


### 3. Dataset References

- **Generation Techniques**  
  Our dataset, comprising 10,000 entries, was created using state-of-the-art generative AI tools, including ChatGPT, Google Bard, and Claude. Each entry included a user situation and a corresponding smart home routine, focusing on diversity, practicality, and contextual relevance.

- **Deduplication and Quality Assurance**  
  Techniques inspired by best practices in dataset cleaning were applied to remove duplicates and ensure the routines aligned with predefined rules. This step maintained dataset quality and enhanced its utility for model training.

- **Structure and Scalability**  
  The dataset was formatted as CSV files, enabling seamless integration with machine learning pipelines. The structured format allowed for scalability and facilitated future expansions using updated prompts or models.


By integrating these tools, methodologies, and datasets, our project bridges the gap between user needs and advanced AI-driven automation, delivering a highly personalized and efficient smart home routine recommender system.

## Conclusion
### Discussion
Our project, *VOICE: A Multimodal Routine Recommender for Any Circumstances*, demonstrates the potential of AI-driven solutions to provide personalized and context-aware smart home automation. By integrating advanced generative AI models with multimodal interaction capabilities, we successfully developed a system that delivers highly practical, actionable, and user-focused routines for diverse situations.

Key highlights include:

- **Innovative Use of Generative AI**: Leveraging state-of-the-art AI models like ChatGPT and paust/pko-chat-t5-large, we created a robust dataset and fine-tuned a model capable of understanding and addressing nuanced user needs.
- **Multimodal Interaction**: Incorporating voice recognition through SKT NUGU and emotional analysis via IBM Watson API enabled us to capture richer user inputs, enhancing the model’s ability to generate contextually relevant outputs.
- **Efficiency and Scalability**: Employing techniques like LoRA for parameter-efficient fine-tuning ensured the system was computationally efficient while maintaining high performance. The structured dataset design allows for seamless scalability and future expansion.

### Closing Remarks
This project highlights the transformative impact of combining generative AI, multimodal interactions, and smart home technologies. Through continuous innovation and user-centered design, we aim to advance smart home automation, creating systems that not only meet but anticipate user needs in a seamless and intuitive manner.
