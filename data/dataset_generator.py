"""
Dataset Generator for Talent Track AI
Generates rich, realistic labeled resume datasets across 10 technical domains,
annotated NER training samples, and paired Resume-Job Description matching records.
"""

import os
import csv
import json
import random

DOMAINS = {
    "Data Science & Machine Learning": {
        "skills": ["Python", "Pandas", "NumPy", "Scikit-Learn", "TensorFlow", "PyTorch", "SQL", "Tableau", "PowerBI", "Matplotlib", "Seaborn", "Jupyter", "Feature Engineering", "Data Cleaning", "NLP", "Machine Learning"],
        "titles": ["Data Scientist", "Machine Learning Engineer", "Data Analyst", "Applied AI Researcher", "ML Ops Specialist"],
        "degrees": ["B.Tech Computer Science", "M.S. Data Science", "B.Sc Statistics", "M.Tech Artificial Intelligence", "B.E. Information Technology"],
        "certs": ["AWS Certified Machine Learning", "TensorFlow Developer Certificate", "IBM Data Science Professional", "DeepLearning.AI Specialization"],
        "phrases": [
            "Developed predictive classification and regression models achieving 94% accuracy.",
            "Conducted exploratory data analysis (EDA) and built interactive dashboards in Tableau.",
            "Designed end-to-end ML pipelines with automated feature selection and hyperparameter tuning.",
            "Implemented deep convolutional and recurrent neural networks using PyTorch and TensorFlow.",
            "Optimized SQL queries across terabyte-scale data lakes for analytical model consumption."
        ]
    },
    "Cloud & DevOps Engineering": {
        "skills": ["Docker", "Kubernetes", "AWS", "Terraform", "CI/CD", "Linux", "Jenkins", "Ansible", "Bash", "Prometheus", "Grafana", "GitLab CI", "Azure", "GCP", "Nginx", "Helm"],
        "titles": ["DevOps Engineer", "Cloud Solutions Architect", "Site Reliability Engineer", "Platform Engineer", "Cloud Infrastructure Specialist"],
        "degrees": ["B.Tech Information Technology", "B.E. Computer Science", "B.Sc Computer Science", "M.Tech Cloud Computing"],
        "certs": ["AWS Certified Solutions Architect", "Certified Kubernetes Administrator (CKA)", "HashiCorp Certified Terraform Associate", "Docker Certified Associate"],
        "phrases": [
            "Architected containerized microservices orchestration on Kubernetes with automated zero-downtime rollouts.",
            "Built robust CI/CD deployment automation pipelines reducing deployment lead time by 65%.",
            "Managed multi-region cloud infrastructure using Terraform Infrastructure as Code (IaC).",
            "Configured real-time cluster telemetry, logging, and alerting with Prometheus and Grafana dashboards.",
            "Hardened Linux production environments and automated configuration management using Ansible."
        ]
    },
    "Full-Stack Web Development": {
        "skills": ["JavaScript", "TypeScript", "React", "Node.js", "Express", "HTML5", "CSS3", "MongoDB", "PostgreSQL", "Next.js", "Redux", "Tailwind CSS", "REST APIs", "GraphQL", "WebSockets"],
        "titles": ["Full Stack Developer", "Software Engineer", "MERN Stack Engineer", "Web Applications Developer", "Lead Frontend & Backend Engineer"],
        "degrees": ["B.Tech Computer Science", "BCA", "MCA", "B.E. Software Engineering", "B.Sc Information Systems"],
        "certs": ["Meta Front-End Developer Certificate", "MongoDB Certified Developer", "AWS Certified Developer - Associate"],
        "phrases": [
            "Engineered responsive high-performance web applications using React.js and modern state management.",
            "Built scalable RESTful APIs and asynchronous backend microservices with Node.js and Express.",
            "Designed normalized PostgreSQL schemas and MongoDB document stores with efficient indexing.",
            "Implemented secure JWT user authentication, role-based access control (RBAC), and webhook integrations.",
            "Improved frontend Core Web Vitals and load times by 45% through code splitting and asset optimization."
        ]
    },
    "Java Backend Development": {
        "skills": ["Java", "Spring Boot", "Spring Cloud", "Microservices", "Hibernate", "JPA", "Kafka", "MySQL", "PostgreSQL", "Redis", "Maven", "Gradle", "JUnit", "Docker", "RabbitMQ"],
        "titles": ["Java Backend Developer", "Senior Java Engineer", "Spring Boot Specialist", "Enterprise Java Architect", "Backend Systems Engineer"],
        "degrees": ["B.Tech Computer Science", "B.E. Computer Engineering", "MCA", "M.Tech Software Engineering"],
        "certs": ["Oracle Certified Professional Java SE", "Spring Certified Professional", "AWS Certified Developer"],
        "phrases": [
            "Engineered enterprise-grade microservices leveraging Spring Boot, Spring Cloud, and Hibernate JPA.",
            "Implemented distributed event-driven architectures utilizing Apache Kafka and RabbitMQ queues.",
            "Optimized multi-threaded backend processing and integrated Redis distributed caching for sub-10ms response times.",
            "Wrote comprehensive unit and integration test suites using JUnit 5 and Mockito achieving 90% code coverage.",
            "Integrated relational SQL databases with Liquibase migrations and resilient connection pooling."
        ]
    },
    "Cyber Security & Network Defense": {
        "skills": ["Network Security", "Penetration Testing", "Wireshark", "Kali Linux", "SIEM", "Splunk", "Metasploit", "Vulnerability Assessment", "Firewalls", "Cryptography", "OWASP", "SOC", "Burp Suite", "Python"],
        "titles": ["Cybersecurity Analyst", "Information Security Specialist", "Penetration Tester", "SOC Analyst", "Security Engineer"],
        "degrees": ["B.Tech Cyber Security", "B.Tech Computer Science", "B.Sc Information Technology", "M.S. Cybersecurity"],
        "certs": ["Certified Ethical Hacker (CEH)", "CompTIA Security+", "CISSP", "Offensive Security Certified Professional (OSCP)"],
        "phrases": [
            "Conducted thorough penetration tests and web application vulnerability assessments based on OWASP Top 10.",
            "Monitored enterprise threat telemetry and investigated security alerts using Splunk SIEM.",
            "Configured firewall rules, intrusion detection systems (IDS/IPS), and zero-trust perimeter network security.",
            "Authored detailed incident response remediation reports and implemented corporate security policies.",
            "Automated threat intelligence log analysis with Python scripting and Wireshark packet capture."
        ]
    },
    "Mobile App Development": {
        "skills": ["Flutter", "Dart", "React Native", "Android", "Kotlin", "Swift", "iOS", "Firebase", "Mobile UI", "REST APIs", "SQLite", "Push Notifications", "App Store Deployment", "Git"],
        "titles": ["Mobile Application Developer", "Flutter Developer", "Android Engineer", "iOS Developer", "Cross-Platform Mobile Engineer"],
        "degrees": ["B.Tech Computer Science", "BCA", "B.E. Information Technology", "MCA"],
        "certs": ["Google Associate Android Developer", "Meta React Native Specialization", "Flutter & Dart Masterclass"],
        "phrases": [
            "Developed native and cross-platform mobile apps for iOS and Android using Flutter and React Native.",
            "Integrated Firebase real-time database, cloud authentication, crashlytics, and push notification services.",
            "Designed fluid 60fps mobile interfaces adhering to Apple Human Interface and Google Material Design guidelines.",
            "Published multiple production apps to Google Play Store and Apple App Store with over 50k downloads.",
            "Implemented offline-first data synchronization with SQLite caching and background sync workers."
        ]
    },
    "AI & Deep Learning Research": {
        "skills": ["Deep Learning", "PyTorch", "TensorFlow", "Transformers", "BERT", "LLMs", "Computer Vision", "OpenCV", "NLP", "Python", "HuggingFace", "CUDA", "LangChain", "GANs"],
        "titles": ["AI Engineer", "Deep Learning Scientist", "Computer Vision Engineer", "NLP Research Specialist", "GenAI Developer"],
        "degrees": ["M.Tech Artificial Intelligence", "Ph.D. Machine Learning", "B.Tech Computer Science", "M.S. Computer Science"],
        "certs": ["NVIDIA Deep Learning Institute", "HuggingFace Certified", "Stanford CS231n / CS224n"],
        "phrases": [
            "Fine-tuned Transformer models (BERT, RoBERTa, LLaMA) for domain-specific NLP extraction and question answering.",
            "Trained object detection and image segmentation networks (YOLOv8, ResNet) with PyTorch and OpenCV.",
            "Engineered Retrieval-Augmented Generation (RAG) pipelines using LangChain, vector databases, and LLM APIs.",
            "Accelerated GPU model inference utilizing TensorRT, ONNX Runtime, and mixed-precision quantization.",
            "Authored and published peer-reviewed AI research on self-supervised representation learning."
        ]
    },
    "Quality Assurance & Test Automation": {
        "skills": ["Selenium", "Cypress", "Python", "Java", "Test Automation", "Postman", "API Testing", "JMeter", "TestNG", "Cucumber", "BDD", "JIRA", "CI/CD Integration", "Performance Testing"],
        "titles": ["QA Automation Engineer", "Software Development Engineer in Test (SDET)", "Lead QA Tester", "Quality Assurance Analyst"],
        "degrees": ["B.Tech Computer Science", "BCA", "B.Sc Computer Science", "B.E. Electronics"],
        "certs": ["ISTQB Certified Tester", "Selenium Automation Specialist", "Postman API Fundamentals Student Expert"],
        "phrases": [
            "Architected scalable UI automation frameworks using Selenium WebDriver and Cypress across multi-browser suites.",
            "Designed and automated REST API regression test suites using Postman, Newman, and PyTest.",
            "Authored behavior-driven development (BDD) test scenarios using Cucumber and Gherkin syntax.",
            "Conducted load and stress testing on backend services using Apache JMeter, pinpointing throughput bottlenecks.",
            "Integrated automated test execution into Jenkins CI/CD pipelines to enforce quality gates on pull requests."
        ]
    },
    "Data Engineering & Big Data": {
        "skills": ["Apache Spark", "Hadoop", "PySpark", "Apache Kafka", "Airflow", "SQL", "Snowflake", "Databricks", "Python", "ETL Pipelines", "AWS S3", "AWS Glue", "Hive", "PostgreSQL"],
        "titles": ["Data Engineer", "Big Data Engineer", "ETL Developer", "Data Platform Architect", "Analytics Engineer"],
        "degrees": ["B.Tech Computer Science", "B.E. Information Technology", "M.S. Data Analytics", "MCA"],
        "certs": ["Databricks Certified Data Engineer", "AWS Certified Data Analytics", "Snowflake SnowPro Core"],
        "phrases": [
            "Designed and orchestrated enterprise batch and streaming ETL pipelines using Apache Spark and Apache Airflow.",
            "Managed scalable data lakehouse architectures on Databricks, Snowflake, and AWS S3.",
            "Engineered real-time data ingestion pipelines from transactional databases into Kafka topics.",
            "Optimized partitioned SQL analytical queries across distributed clusters, reducing job execution times by 50%.",
            "Implemented automated data quality checks, schema evolution validations, and lineage tracking."
        ]
    },
    "UI/UX Design & Frontend Engineering": {
        "skills": ["Figma", "UI/UX Design", "Wireframing", "Prototyping", "User Research", "HTML5", "CSS3", "JavaScript", "React", "Design Systems", "Adobe XD", "Accessibility (a11y)", "Interaction Design"],
        "titles": ["UI/UX Designer", "Product Designer", "Frontend UX Engineer", "Interaction Designer", "Design Systems Lead"],
        "degrees": ["Bachelor of Design (B.Des)", "B.Tech Computer Science", "B.Sc Multimedia & Animation", "B.A. Graphic Design"],
        "certs": ["Google UX Design Professional Certificate", "Interaction Design Foundation (IxDF)", "Nielsen Norman Group UX Master"],
        "phrases": [
            "Created comprehensive high-fidelity interactive prototypes and atomic design systems in Figma.",
            "Conducted usability testing sessions and translated user feedback into wireframes and streamlined customer journeys.",
            "Collaborated closely with engineering to translate Figma component specs into accessible HTML5/CSS3 and React components.",
            "Established WCAG 2.1 AA accessibility compliance standards across responsive mobile and web interfaces.",
            "Conducted quantitative and qualitative user research, persona creation, and journey mapping."
        ]
    }
}

FIRST_NAMES = ["Aarav", "Priya", "Rahul", "Ananya", "Rohan", "Sneha", "Vikram", "Neha", "Aditya", "Pooja", "Arjun", "Kavya", "Siddharth", "Ishita", "Amit", "Divya", "Karan", "Rhea", "Manish", "Tanvi", "Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "David", "Emma", "Daniel"]
LAST_NAMES = ["Sharma", "Verma", "Patel", "Gupta", "Singh", "Kumar", "Iyer", "Nair", "Reddy", "Choudhury", "Bose", "Mehta", "Deshmukh", "Joshi", "Mishra", "Smith", "Johnson", "Williams", "Brown", "Miller", "Davis", "Wilson", "Anderson", "Thomas", "Taylor"]
COLLEGES = ["National Institute of Technology", "Indian Institute of Technology", "BITS Pilani", "Delhi Technological University", "Vellore Institute of Technology", "Anna University", "Mumbai University", "Pune Institute of Computer Technology", "Manipal Institute of Technology", "SRM Institute of Science & Technology"]


def generate_single_resume(domain_name: str, candidate_id: int) -> dict:
    domain_info = DOMAINS[domain_name]
    
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    email = f"{first.lower()}.{last.lower()}{random.randint(10, 999)}@example.com"
    phone = f"+91 {random.randint(70000, 99999)} {random.randint(10000, 99999)}"
    
    title = random.choice(domain_info["titles"])
    degree = random.choice(domain_info["degrees"])
    college = random.choice(COLLEGES)
    grad_year = random.randint(2017, 2024)
    
    yoe = round(random.uniform(0.5, 9.5), 1)
    
    # Pick 5 to 9 skills
    sample_skills = random.sample(domain_info["skills"], min(len(domain_info["skills"]), random.randint(5, 9)))
    
    # Optional certification
    has_cert = random.random() > 0.3
    cert = random.choice(domain_info["certs"]) if has_cert else "None"
    
    # Pick 2 to 3 experience phrases
    exp_bullets = random.sample(domain_info["phrases"], random.randint(2, 3))
    
    # Construct complete realistic resume text
    resume_text = f"""
{name.upper()}
{title} | {email} | {phone} | LinkedIn: linkedin.com/in/{first.lower()}-{last.lower()}

PROFESSIONAL SUMMARY
Results-driven {title} with {yoe} years of hands-on experience in {domain_name}. Proven expertise in {', '.join(sample_skills[:4])}. Adept at designing scalable solutions, driving team efficiency, and delivering high-impact technological innovations.

CORE SKILLS & TECHNOLOGIES
{', '.join(sample_skills)}

PROFESSIONAL EXPERIENCE
{title} — Tech Solutions Corp ({2024 - int(yoe)} - Present)
• {exp_bullets[0]}
• {exp_bullets[1]}
{('• ' + exp_bullets[2]) if len(exp_bullets) > 2 else ''}

EDUCATION
{degree} — {college} (Graduated {grad_year})
CGPA: {round(random.uniform(7.0, 9.8), 2)} / 10.0

CERTIFICATIONS & ACHIEVEMENTS
{f'• {cert}' if has_cert else '• Active Contributor to Open Source Projects'}
• Recognized for outstanding technical contribution and sprint delivery excellence.
""".strip()

    return {
        "id": f"RES-{candidate_id:04d}",
        "name": name,
        "email": email,
        "domain": domain_name,
        "title": title,
        "yoe": yoe,
        "skills": "; ".join(sample_skills),
        "degree": degree,
        "college": college,
        "certification": cert,
        "resume_text": resume_text
    }


def generate_all_datasets(output_dir: str, num_samples_per_domain: int = 110):
    os.makedirs(output_dir, exist_ok=True)
    
    all_resumes = []
    cand_id = 1
    
    print(f"Generating multi-domain resume dataset across {len(DOMAINS)} domains...")
    for domain_name in DOMAINS.keys():
        for _ in range(num_samples_per_domain):
            res = generate_single_resume(domain_name, cand_id)
            all_resumes.append(res)
            cand_id += 1
            
    random.shuffle(all_resumes)
    
    # 1. Save Resume Dataset CSV
    csv_path = os.path.join(output_dir, "resume_dataset.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_resumes[0].keys()))
        writer.writeheader()
        writer.writerows(all_resumes)
    print(f"-> Saved {len(all_resumes)} resume records to {csv_path}")

    # 2. Generate Semantic Matching Pairs (Resume vs JD)
    print("Generating Resume-to-Job matching pairs dataset...")
    pairs = []
    for i, res in enumerate(all_resumes[:500]):
        domain = res["domain"]
        domain_skills = DOMAINS[domain]["skills"]
        
        # Scenario A: Exact Domain Match (High score: 75 - 98)
        jd_match = f"""Looking for a senior {DOMAINS[domain]["titles"][0]} to join our growing product team. 
Must be proficient in {', '.join(random.sample(domain_skills, 4))}. Minimum 3+ years of professional industry experience.
Strong foundation in system architecture, microservices, and modern collaborative engineering methodologies."""
        pairs.append({
            "pair_id": f"PAIR-{i*2+1:04d}",
            "resume_id": res["id"],
            "resume_domain": domain,
            "target_role": DOMAINS[domain]["titles"][0],
            "resume_text": res["resume_text"],
            "job_description": jd_match,
            "is_matched_domain": 1,
            "ground_truth_score": round(random.uniform(78.0, 96.0), 1)
        })
        
        # Scenario B: Cross-Domain / Low Match (Low score: 25 - 55)
        diff_domains = [d for d in DOMAINS.keys() if d != domain]
        diff_domain = random.choice(diff_domains)
        diff_skills = DOMAINS[diff_domain]["skills"]
        jd_mismatch = f"""Hiring a specialist for {DOMAINS[diff_domain]["titles"][0]}.
Primary requirements include hands-on experience with {', '.join(random.sample(diff_skills, 4))}.
Candidate will lead our {diff_domain} initiatives and deploy resilient enterprise pipelines."""
        pairs.append({
            "pair_id": f"PAIR-{i*2+2:04d}",
            "resume_id": res["id"],
            "resume_domain": domain,
            "target_role": DOMAINS[diff_domain]["titles"][0],
            "resume_text": res["resume_text"],
            "job_description": jd_mismatch,
            "is_matched_domain": 0,
            "ground_truth_score": round(random.uniform(28.0, 52.0), 1)
        })

    random.shuffle(pairs)
    pairs_csv = os.path.join(output_dir, "matching_pairs.csv")
    with open(pairs_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        writer.writeheader()
        writer.writerows(pairs)
    print(f"-> Saved {len(pairs)} matching pairs to {pairs_csv}")

    # 3. Generate Annotated NER Training Data
    print("Generating custom NER span annotations...")
    ner_samples = []
    for res in all_resumes[:300]:
        text = res["resume_text"]
        entities = []
        
        # Name span
        name = res["name"]
        idx = text.find(name.upper())
        if idx != -1:
            entities.append([idx, idx + len(name), "PERSON"])
            
        # Title span
        title = res["title"]
        idx = text.find(title)
        if idx != -1:
            entities.append([idx, idx + len(title), "DESIGNATION"])
            
        # Degree span
        degree = res["degree"]
        idx = text.find(degree)
        if idx != -1:
            entities.append([idx, idx + len(degree), "DEGREE"])
            
        # Skills spans
        skills_list = [s.strip() for s in res["skills"].split(";")]
        for s in skills_list:
            pos = 0
            while True:
                idx = text.find(s, pos)
                if idx == -1:
                    break
                entities.append([idx, idx + len(s), "SKILL"])
                pos = idx + len(s)
                
        ner_samples.append({
            "text": text,
            "entities": entities
        })
        
    ner_json = os.path.join(output_dir, "ner_training_data.json")
    with open(ner_json, mode="w", encoding="utf-8") as f:
        json.dump(ner_samples, f, indent=2)
    print(f"-> Saved {len(ner_samples)} annotated NER samples to {ner_json}")

    print("Data generation finished successfully!")
    return {
        "resumes_count": len(all_resumes),
        "pairs_count": len(pairs),
        "ner_count": len(ner_samples)
    }


if __name__ == "__main__":
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    generate_all_datasets(target_dir, num_samples_per_domain=110)
