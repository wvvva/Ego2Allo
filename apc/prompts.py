# ---------------------------------------------------------------- # 
# Predefined prompt templates
# ---------------------------------------------------------------- # 

# Main prompt for 'get_objects_of_interest'
PROMPT_GET_OBJECTS_OF_INTEREST = """
### Situation Description
Given an image and a spatial reasoning question, we need to all entities that are included in the question.

# Example 1
[Question] You are standing at the airplane's position, facing where it is facing. Is the the person on your left or right?
[Detect] [airplane, person]

# Examples 2
[Question] From the old man's perspective, is the person wearing a hat on the left of the green car?
[Detect] [old man, person wearing a hat, green car]

# Examples 3
[Question] From the car's perspective, which is on the right side: the person or the tree?
[Detect] [car, person, tree]

### Your Task
Now, given the question below, please identify the entities that are included in the question.

[Question] {question}
[Detect]
"""

# Auxiliary prompt for when the VLM fails to match the format for 'get_objects_of_interest'
PROMPT_GET_OBJECTS_OF_INTEREST_AUX = """
Looks like your response is not in the correct format!

Previous response: {response}

Please modify your response to the correct format.

[Question] {question}
[Detect]
"""

# Pattern to match for 'get_objects_of_interest'
PATTERN_GET_OBJECTS_OF_INTEREST = r"\[[^\]]*\]"

# Prompt for 'get_reference_viewer'
PROMPT_GET_REFERENCE_VIEWER = """
Given a question about spatial reasoning, we want to extract the **perspective** of the question.

If the question is from the camera's perspective, return ++camera++.

### Example 1
[Question] From the camera's perspective, which side of the car is facing the camera?
[Perspective] ++camera++

### Example 2
[Question] From the woman's perspective, is the tree on the left or right?
[Perspective] ++woman++

### Example 3
[Question] If you are standing at the airplane's position, facing where it is facing, is the car on your left or right?
[Perspective] ++airplane++

### Your Task
Given the question below, please specify the **perspective** from which the question is asked.
You must return in the format: [Perspective] ++object_name++

[Question] {question}
[Options] {obj_str}, camera

[Perspective]
"""

# Pattern to match for 'get_reference_viewer'
PATTERN_GET_REFERENCE_VIEWER = r"\+\+(.*?)\+\+"

# Prompt for 'convert_to_ego' - remove perspective description
PROMPT_CONVERT_TO_EGO = """
From a sentence with a perspective description, we need to remove the perspective description.

# Example 1
[Question] From the car's perspective, which is on the right side: blue cube or the green cube?
[Reference Viewer] car
[Output] Which is on the right side: blue cube or the green cube?

# Example 2
[Question] You are standing at the airplane's position, facing where it is facing. Is the the red cube on your left or right?
[Reference Viewer] airplane
[Output] Is the the red cube on your left or right?

# Example 3
[Question] From the cat's viewpoint, is the yellow cube wearing a hat on the left of the green cube?
[Reference Viewer] cat
[Output] Is the yellow cube wearing a hat on the left of the green cube?

# Your Task
Given the question below, please remove the perspective description.
[Question] {question}
[Reference Viewer] {ref_viewer}
[Output]
"""

PROMPT_PERSPECTIVE_VISUAL = """
This is an image of a 3D scene. 

- The viewer is facing towards the object that is **closest to the center**.
- A **larger** object is closer to the viewer compared to a **smaller** object.

# Task
Based on the image, please answer the following question.

{question}

Answer the question in the format:
[Reasoning] <reasoning>
[Answer] <answer>
"""

# Prompt for 'prompt_abstract_to_real'
PROMPT_ABSTRACT_TO_REAL = """
We provide an color-object map that maps each colored box to an object.

# Color-Object Map
{color_obj_map}

# Task
Given the color-object map below, change all the colored boxes in the below response into their respective objects.

[Original Response]
{abstract_response}

[Real Response]
"""

# Auxiliary prompt for choosing options for evaluation
PROMPT_CHOOSE_OPTIONS = """
Here is the question and the response you provided:

[Question] {question}
[Response] {response}

Based on the response you provided, please choose the correct option from the following list:
[Options] {options}
Return only the option
"""

# ---------------------------------------------------------------- # 

class PromptParser:
    def __init__(self, config):
        self.config = config
        # Define predefined prompts for different use cases
        self.predefined_prompts = {
            "get_objects_of_interest": PROMPT_GET_OBJECTS_OF_INTEREST,
            "get_objects_of_interest_aux": PROMPT_GET_OBJECTS_OF_INTEREST_AUX,
            "pattern_get_objects_of_interest": PATTERN_GET_OBJECTS_OF_INTEREST,
            "get_reference_viewer": PROMPT_GET_REFERENCE_VIEWER,
            "pattern_reference_viewer": PATTERN_GET_REFERENCE_VIEWER,
            "convert_to_ego": PROMPT_CONVERT_TO_EGO,
            "perspective_visual": PROMPT_PERSPECTIVE_VISUAL,
            "abstract_to_real": PROMPT_ABSTRACT_TO_REAL,
            "choose_options": PROMPT_CHOOSE_OPTIONS,
        }

    # def parse_prompt(self, prompt):
    #     return prompt

    def get_prompt_by_type(self, prompt_type: str) -> str:
        """
        Router function that returns predefined prompts based on the prompt type.
        
        Args:
            prompt_type (str): The type of prompt to retrieve
            
        Returns:
            str: The predefined prompt for the given type
            
        Raises:
            KeyError: If the prompt_type is not found in predefined_prompts
        """
        if prompt_type not in self.predefined_prompts:
            raise KeyError(f"Unknown prompt type: {prompt_type}. Available types: {list(self.predefined_prompts.keys())}")
        
        return self.predefined_prompts[prompt_type]

    def list_available_prompts(self) -> list:
        """
        Return a list of all available prompt types.
        
        Returns:
            list: List of available prompt type names
        """
        return list(self.predefined_prompts.keys())

