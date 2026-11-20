def generate_recommendation(row, prediction):
    """Generate transparent, rule-based learning recommendations."""
    recommendations = []

    def get_number(column):
        try:
            return float(row[column])
        except (KeyError, TypeError, ValueError):
            return None

    study_time = get_number("studytime")
    failures = get_number("failures")
    absences = get_number("absences")

    if study_time is not None and study_time <= 2:
        recommendations.append(
            "Increase weekly study time and follow a regular timetable."
        )

    if failures is not None and failures > 0:
        recommendations.append(
            "Attend additional academic support and revision sessions."
        )

    if absences is not None and absences > 10:
        recommendations.append(
            "Improve attendance and participate in classroom activities."
        )

    if prediction == "At Risk":
        recommendations.append(
            "Arrange additional teacher guidance and monitor progress."
        )
    elif prediction == "On Track":
        recommendations.append(
            "Continue regular revision and attempt advanced exercises."
        )

    if not recommendations:
        recommendations.append(
            "Maintain consistent learning habits and regular practice."
        )

    return recommendations
