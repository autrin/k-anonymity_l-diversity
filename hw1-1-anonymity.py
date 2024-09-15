"""
The key challenge here is anonymizing the dataset to protect users' privacy (based on their salary) while maintaining as much utility of the dataset as possible. The two levels of anonymity are:

    k1 = 10 for users with salaries ≤ 50K (stronger privacy).
    k2 = 5 for users with salaries > 50K (less strict privacy).

QIs:
    The four attributes I need to anonymize through generalization or suppression are:

    Age: Numerical attribute.
    Education: Categorical attribute with multiple levels (e.g., Bachelors, HS-grad).
    Marital-Status: Categorical attribute with different marital statuses.
    Race: Categorical attribute with 5 distinct values (White, Asian-Pac-Islander, etc.).

Sensitive Attribute: The occupation is treated as sensitive and must remain in the dataset. 
However, you need to ensure that the anonymized data prevents users from being easily identified based on this attribute.


"""