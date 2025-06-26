import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\DA_project_CSV_FILES\StudentsPerformance.csv')

print("\n📋 First 5 rows:")
print(df.head())
      
print("\nℹ️ Dataset Info:")
print(df.info())

print("\n📊 Summary:")
print(df.describe())

print("\n🧼 Missing values:")
print(df.isnull().sum())

print("\n📈 Average scores by gender:")
print(df.groupby('gender')[['math score', 'reading score', 'writing score']].mean())

plt.figure(figsize=(8, 5))
sns.boxplot(x='gender', y='math score', data=df)
plt.title('Math Score by Gender')
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x='test preparation course', y='math score', data=df)
plt.title('Math Score vs Test Preparation')
plt.show()

df['average score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
plt.figure(figsize=(10, 6))
sns.barplot(x='parental level of education', y='average score', data=df, palette='viridis')
plt.xticks(rotation=45)
plt.title('Average Score by Parental Education')
plt.show()

df.to_csv('D:/student_project/cleaned_student_data.csv', index=False)
print("\n✅ Analysis complete! Cleaned file saved.")
