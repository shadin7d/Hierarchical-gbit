# Hierarchical G-BIT Research Framework

class GBITFramework:
    def __init__(self, name):
        self.name = name
        self.subsystems = []

    def add_subsystem(self, subsystem):
        self.subsystems.append(subsystem)

    def get_summary(self):
        summary = f"Framework: {self.name}\n"
        for subsystem in self.subsystems:
            summary += f"- {subsystem.get_summary()}\n"
        return summary

class Subsystem:
    def __init__(self, name):
        self.name = name

    def get_summary(self):
        return f"Subsystem: {self.name}"

# Example usage
if __name__ == '__main__':
    main_framework = GBITFramework("Main G-BIT Framework")
    subsystem1 = Subsystem("Subsystem A")
    subsystem2 = Subsystem("Subsystem B")
    main_framework.add_subsystem(subsystem1)
    main_framework.add_subsystem(subsystem2)

    print(main_framework.get_summary())