import logging
class PromoAnalysis:
    def __init__(self):
        """
        Initialize Promotoin Analysis Class.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized Promotoin Sales Analysis class.")
    def analyze_promo_effect_on_sales(self, df):
        self.logger.info("Starting to promotion Analysis on Sales")
        promo_sales = df[df['Promo'] == 1]['Sales'].mean()
        non_promo_sales = df[df['Promo'] == 0]['Sales'].mean()
        
        print(f"Average Sales with Promo: {promo_sales:.2f}")
        print(f"Average Sales without Promo: {non_promo_sales:.2f}")
        self.logger.info("Successfully Checking promotion Analysis on sales")
        
        return promo_sales, non_promo_sales
    def analyze_promo_effect_on_customers(self, df):
        self.logger.info("Starting to promotion Analysis on Customers")
        promo_customers = df[df['Promo'] == 1]['Customers'].mean()
        non_promo_customers = df[df['Promo'] == 0]['Customers'].mean()
        
        print(f"Average Customers with Promo: {promo_customers:.2f}")
        print(f"Average Customers without Promo: {non_promo_customers:.2f}")
        self.logger.info("Successfully Checking promotion Analysis on Customers")
        return promo_customers, non_promo_customers
    def analyze_sales_per_customer(self, df):
        self.logger.info("Starting to promotion Analysis on sales vs Customers")
        df['Sales_per_Customer'] = df['Sales'] / df['Customers']
        promo_spc = df[df['Promo'] == 1]['Sales_per_Customer'].mean()
        non_promo_spc = df[df['Promo'] == 0]['Sales_per_Customer'].mean()
        
        print(f"Sales per Customer with Promo: {promo_spc:.2f}")
        print(f"Sales per Customer without Promo: {non_promo_spc:.2f}")
        self.logger.info("Successfully Checking promotion Analysis Sales vs Customers")
        return promo_spc, non_promo_spc