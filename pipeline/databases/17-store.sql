-- SQL script for 17-store.sql
CREATE TRIGGER decrease_quantity_after_order
AFTER INSERT ON orders
FOR EACH ROW
    UPDATE items
    SET quantity = quantity - NEW.number
    WHERE name = NEW.item_name;