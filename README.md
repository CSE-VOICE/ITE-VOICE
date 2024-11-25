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

**1.	Prompt Design:**
   
We crafted a detailed prompt designed to elicit smart home routines that directly control appliances, avoiding actions requiring human intervention. This ensured that the dataset aligns with practical use cases for smart home automation. The prompt included:
+ A predefined list of smart home devices and their actionable settings.
+ Explicit guidelines for valid and invalid routines.
+	Examples of diverse scenarios to encourage variety and creativity.
  
**2.	Generative AI Utilization:**
   
Generative AI models were used to produce routines in a structured JSON format containing:
+	Situation: A natural language description of the context, including the time and specific details.
+	Routine: A concise description of device actions adhering to the predefined rules.

**3.	Deduplication:**
   
To ensure the uniqueness of the dataset, we implemented a deduplication process to remove identical situations and their corresponding routines. This step helped maintain variety while preventing redundancy.

**4.	Quality Assurance:**
 
The generated routines were filtered and reviewed to ensure compliance with the following criteria:
+	Practical and actionable by smart home devices.
+	Free from human-performed actions (e.g., “hang laundry”).
+	Well-aligned with the context described in the situation.

**5.	Data Export to CSV:**

Once the routines satisfied all the criteria outlined in the previous steps, they were added to a structured CSV file for easier handling and further processing. This step involved:
+ Transforming the JSON-formatted routines into a tabular CSV structure with columns for “Situation” and “Routine.”
+ Ensuring proper formatting and compatibility for downstream applications, such as training machine learning models or integrating with smart home systems.
+ Storing the resulting dataset as dataset.csv, which serves as the foundation for subsequent analysis and model training.

**Dataset Highlights**

+	Diversity: The dataset covers a wide range of situations, including both mundane and imaginative contexts. Examples range from “빨래가 너무 많이 쌓였어, 세탁기 돌려야 돼.” to “집안에 벌레가 들어왔어, 조명을 이용해서 유인해보자.”
+	Structure: Each entry consists of a situation and a corresponding routine in CSV format, enabling seamless integration into machine learning pipelines.
+	Scalability: The methodology allows for further expansion by generating additional routines using updated prompts or new AI models.

Sample Entry

```
"situation": "집에 손님이 오기 전에 분위기를 밝게 만들고 싶어.",
"routine": "에어컨을 23도로 설정하고 조명을 밝게 조정하며 공기청정기를 켤게요."
```

Generated Dataset Statistics

	• Total routines requested: 15,000
	• Unique routines after deduplication: 10,000
	• Eliminated duplicates & quality checking: Approximately 50% of initial attempts

This dataset represents a cutting-edge resource for training and fine-tuning AI systems designed to enhance smart home automation, providing realistic and user-centered automation routines tailored to a wide array of situations.

## Methodology
**1. Data Preprocessing**

To ensure consistent and high-quality input for training the model, we conducted thorough preprocessing of the dataset. The primary goal was to standardize the input situation text by embedding it into a predefined template. This helped the model better understand the context and generate consistent and accurate outputs.

+	Input Template Definition:
  We defined an input template that outlined specific rules for appliance actions and included examples of both valid and invalid routines. The template     ensured that:	
    + Only device-controllable actions were included.
    +	Human-performed actions were excluded.
    +	Device settings were specific and actionable.

+	Implementation:

Using a Python script, each situation in the dataset was transformed using the template, and the preprocessed data was saved in a new CSV file (preprocessed_dataset.csv). 

**2. Fine-Tuning the Model**

We fine-tuned a pre-trained transformer model to align it with the unique requirements of our smart home automation service.

+	Model Selection:

We used the **paust/pko-chat-t5-large model**, a transformer-based variant of Google Flan-T5 fine-tuned with Korean chat data, as the base model. This choice was made due to its conversational capabilities and suitability for Korean-language tasks.

```
# Load Tokenizer & Model
tokenizer = T5TokenizerFast.from_pretrained('paust/pko-chat-t5-large')
model = T5ForConditionalGeneration.from_pretrained('paust/pko-chat-t5-large')
model.to(device)
```

+	Challenges and Solutions:
    +	Large Model Size: The model, with over 800M parameters, posed computational challenges for our A10 24GB GPU. To address this, we used the PEFT-based LoRA (Low-Rank Adaptation) approach to reduce the number of trainable parameters while maintaining model performance.
    +	Memory Optimization: The model was loaded with bfloat16 precision, reducing memory usage without significant loss in performance.
  
+	LoRA Configuration:

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

**3. Training Procedure**

We trained the model using the preprocessed dataset in a seq2seq learning framework:

+	Data Splitting: The dataset was split into training (95%) and validation (5%) sets to evaluate model performance during training.

+ Hyperparameters:
    + Optimizer: AdamW
    +	Learning Rate: 1e-5
    +	Batch Size: 4
    +	Epochs: 10    
  
+	Checkpoints:
    +	Checkpoints were saved every 0.5 epoch to ensure robustness against interruptions and to allow iterative evaluation.
    +	The final model was selected based on validation performance from the saved checkpoints.

```
# save checkpoint for every 0.5 epoch
if total_steps % int(len(train_dataset) / (batch_size * 2)) == 0:
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{total_steps}")
    model.save_pretrained(checkpoint_path)
```

+	Training Process:

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

**4. Performance Monitoring**

Throughout the training process:

+	Training Loss: Monitored at regular intervals (every 100 steps (log_interval)) to ensure convergence.
+	Validation Loss: Evaluated after each epoch to track generalization and avoid overfitting.

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

**5. Final Model Selection**

After training, the final model was selected based on its validation performance from the last checkpoint. The model was saved for deployment, capable of generating optimized routines for smart home devices based on the input situations.

This methodology ensured that the resulting model was not only computationally efficient but also highly tailored to the specific requirements of the service, offering accurate and actionable routines for various scenarios.

## Evaluation & Analysis
tbd

## Related Work
tbd

## Conclusion
tbd
